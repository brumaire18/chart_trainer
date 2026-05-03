# Pair Trading Daily Recommendation

## Trade Config

- lookback: `60`
- entry_z: `1.75`
- exit_z: `0.5`
- stop_z: `3.5`
- max_holding_days: `20`

## Summary

- candidate_count: `12`
- recommendation_count: `8`

## Recommendations

| pair | signal_date | signal_state | action | current_z | prev_z | z_delta_1d | distance_to_entry | distance_to_exit | distance_to_stop | half_life | timing_note | event_warning | event_disclosure_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18020-18030 | 2026-05-01 | entry_short_spread | sell_y_buy_x | 2.023914 | 2.431685 | -0.407771 | 0.0 | 1.523914 | 1.476086 | 7.647528 | z が entry_z を上回っています。Y売り / X買いの新規候補です。 |  |  |
| 52680-52820 | 2026-05-01 | entry_long_spread | buy_y_sell_x | -2.105196 | -2.035804 | -0.069392 | 0.0 | 1.605196 | 1.394804 | 7.681382 | z が -entry_z を下回っています。Y買い / X売りの新規候補です。 |  |  |
| 83060-84110 | 2026-05-01 | take_profit_zone | take_profit_exit | 0.373246 | -0.049281 | 0.422527 | 1.376754 | 0.0 | 3.126754 | 11.791381 | z が exit_z の内側へ戻っており、利確・手仕舞いを優先する局面です。 |  |  |
| 83160-84110 | 2026-05-01 | take_profit_zone | take_profit_exit | -0.255367 | 0.192344 | -0.44771 | 1.494633 | 0.0 | 3.244633 | 11.119323 | z が exit_z の内側へ戻っており、利確・手仕舞いを優先する局面です。 |  |  |
| 31160-59490 | 2026-05-01 | take_profit_zone | take_profit_exit | 0.093454 | 0.46976 | -0.376306 | 1.656546 | 0.0 | 3.406546 | 17.250184 | z が exit_z の内側へ戻っており、利確・手仕舞いを優先する局面です。 |  |  |
| 94330-94340 | 2026-05-01 | watch | watch | 1.586617 | 1.966523 | -0.379906 | 0.163383 | 1.086617 | 1.913383 | 7.999353 | z は平均回帰方向へ進んでいます。反対売買のタイミングを待つ局面です。 |  |  |
| 80530-80580 | 2026-05-01 | watch | watch | 0.899468 | -0.264201 | 1.163669 | 0.850532 | 0.399468 | 2.600532 | 15.223855 | z はさらに乖離方向へ進んでいます。新規候補でも条件確認を優先します。 |  |  |
| 34070-41830 | 2026-05-01 | watch | watch | -0.638372 | -0.735588 | 0.097215 | 1.111628 | 0.138372 | 2.861628 | 7.796236 | z は平均回帰方向へ進んでいます。反対売買のタイミングを待つ局面です。 |  |  |

## Exit Timing Guide

- `abs(z) <= exit_z`: 利確・手仕舞いを優先
- `abs(z) >= stop_z`: 損切りを優先
- `z_delta_1d < 0` かつ `abs(z)` 縮小: 平均回帰が進行中
- `z_delta_1d > 0` かつ `abs(z)` 拡大: 乖離拡大に注意
- `event_warning`: 決算や開示でスプレッドが歪んでいる可能性に注意