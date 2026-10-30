"""Tests for economically consequential alignment and leakage invariants."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
DATA=REPO/'data/processed/wti_daily.csv'
DICT=REPO/'data/processed/features.csv'
sys.path.insert(0,str(ROOT/'src'))
from oil_lstm import (read_data, origins_between, windows, targets, fit_scaler,
                      boundary, forecast_from_checkpoint, seed_everything,
                      published_origins, target_calendar, forecast_checkpoint_frame,
                      OilLSTM, PROTOCOL_ID, FIT_END, TEST_START)
from unittest.mock import patch
import json
import torch


class CausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(42)
        cls.df,cls.cols=read_data(DATA,DICT)

    def test_no_target_in_predictors(self):
        self.assertNotIn('target_wti',self.cols)
        self.assertEqual(len(self.cols),32)

    def test_all_horizons_respect_training_boundary(self):
        end=boundary(self.df,'2023-08-31',0)
        ii=published_origins(self.df,origins_between(len(self.df),60,0,end),'2023-08-31')
        self.assertTrue(np.all(ii+60<=end))
        from m3_ml.common.data import label_available_dates
        availability=label_available_dates(self.df)
        self.assertTrue(all((availability.iloc[i+1:i+61]<=pd.Timestamp('2023-08-31')).all() for i in ii))

    def test_window_ends_at_origin(self):
        x=np.arange(300).reshape(100,3)
        np.testing.assert_array_equal(windows(x,[70],20)[0],x[51:71])
        altered=x.copy();altered[71:]=-999
        np.testing.assert_array_equal(windows(x,[70],20),windows(altered,[70],20))

    def test_target_horizon_one_is_next_quoted_row(self):
        delta=targets(self.df,[100])[0]
        anchor=self.df.wti_spot_last_published.iloc[100]
        self.assertAlmostEqual(delta[0]+anchor,self.df.target_wti.iloc[101])
        self.assertAlmostEqual(delta[59]+anchor,self.df.target_wti.iloc[160])

    def test_negative_price_is_retained(self):
        i=int(np.argmin(self.df.target_wti.to_numpy()))
        self.assertAlmostEqual(self.df.target_wti.iloc[i],-36.98)
        self.assertTrue(np.isfinite(targets(self.df,[i])).all())

    def test_constant_scaler_does_not_divide_by_zero(self):
        mu,sd=fit_scaler(np.ones((8,3)))
        np.testing.assert_array_equal((np.ones((8,3))-mu)/sd,np.zeros((8,3)))

    def test_forecast_is_independent_of_all_targets(self):
        checkpoint=ROOT/'outputs/frozen_october_2026/models'
        if not (checkpoint/'metadata.json').exists():
            self.skipTest('Train model before checkpoint integration test')
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as td:
            td=Path(td)
            baseline=forecast_from_checkpoint(DATA,DICT,checkpoint,td/'a.csv')
            unlabeled=self.df.copy();unlabeled['target_wti']=np.nan
            unlabeled.to_csv(td/'unlabeled.csv',index=False)
            actual=forecast_from_checkpoint(td/'unlabeled.csv',DICT,checkpoint,td/'b.csv')
            np.testing.assert_allclose(actual.prediction,baseline.prediction,rtol=0,atol=1e-10)

    def test_unknown_calendar_is_not_invented(self):
        dates,observed=target_calendar(self.df,len(self.df)-1)
        self.assertEqual(observed,1)
        self.assertTrue(all(pd.isna(d) for d in dates[1:]))

    def test_publication_purge_checks_every_label(self):
        frame=pd.DataFrame({'date':pd.bdate_range('2020-01-01',periods=100)})
        available=frame.date.copy()
        available.iloc[30]=pd.Timestamp('2030-01-01')
        with patch('oil_lstm.label_available_dates',return_value=available):
            result=published_origins(frame,np.array([0,31,39]),'2021-01-01')
        np.testing.assert_array_equal(result,[31,39])

    def test_fixed_path_ignores_later_october_predictors(self):
        # Random weights exercise inference causality only, not a trained result.
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            cfg=dict(lookback=20,hidden=4,dropout=0.0)
            metadata=dict(protocol_id=PROTOCOL_ID,fit_label_end=FIT_END,
                          config=cfg,features=self.cols,seeds=[42],anchor_policy="scheduled_quote_asof")
            (folder/'metadata.json').write_text(json.dumps(metadata))
            np.savez(folder/'scalers.npz',x_mean=np.zeros(len(self.cols)),
                     x_std=np.ones(len(self.cols)),y_mean=np.zeros(60),y_std=np.ones(60))
            torch.save(OilLSTM(len(self.cols),4,0.0).state_dict(),folder/'lstm_seed_42.pt')
            original=forecast_checkpoint_frame(self.df,self.cols,folder,modes=('fixed_origin',))
            altered=self.df.copy()
            first=int(altered.date.searchsorted(pd.Timestamp(TEST_START)))
            altered.loc[first:,self.cols]=-999.0
            altered['target_wti']=np.nan
            result=forecast_checkpoint_frame(altered,self.cols,folder,modes=('fixed_origin',))
            np.testing.assert_array_equal(original.forecast,result.forecast)
            self.assertTrue(result.actual.isna().all())
            self.assertTrue((result.origin_date==pd.Timestamp(FIT_END)).all())


if __name__=='__main__':
    unittest.main(verbosity=2)
