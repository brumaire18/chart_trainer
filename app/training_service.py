# app/training_service.py

from dataclasses import dataclass
from typing import Tuple
import pandas as pd


@dataclass
class TrainingSession:
    """
    1問分の練習セッション情報。
    """
    symbol: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    answer_end_date: pd.Timestamp


def create_random_session(
    df_all: pd.DataFrame,
    symbol: str,
    lookback_bars: int,
    answer_bars: int,
) -> Tuple[TrainingSession, pd.DataFrame, pd.DataFrame]:
    """
    全期間の株価DataFrameから、練習用の区間をランダムに1つ切り出す。

    戻り値:
      - TrainingSession: メタ情報
      - df_problem: 問題部分 (lookback_bars 本)
      - df_answer: 答え部分 (answer_bars 本)
    """
    raise NotImplementedError
