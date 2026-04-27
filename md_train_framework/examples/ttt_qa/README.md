# TicTacToe QA

Client-facing query example for weighted multi-task tic-tac-toe reasoning.

Defaults:
- dataset: `maxs-m87/tictactoe-qa-v1`
- backend profile: `ttt_query`
- reward preset: `query_ranked_reward`
- selection metric: `eval_reward_mean`
- best-move reward mode: `ranked`
- task mix matches the legacy `v1` sweep family: `best_move=3`, `available_moves_count=3`, `available_moves_list=4`
- reasoning enabled by default
- off-policy replay enabled
- staging API: `https://api-staging.moondream.ai/v1`
- env file: repo-root `.env.staging`

Commands:

```bash
python -m md_train_framework.examples.ttt_qa.dataset_loader
python -m md_train_framework.examples.ttt_qa.train
python -m md_train_framework.examples.ttt_qa.eval --finetune-id YOUR_FINETUNE_ID --checkpoint-step YOUR_STEP
python -m md_train_framework.examples.ttt_qa.sweep --plan-only
python -m md_train_framework.examples.ttt_qa.leaderboard
```

The default config now follows the strongest in-repo `tictactoe-qa-v1` sweep family rather than the earlier best-move-only showcase bias. The profile still owns task weighting, task-specific token caps, ranked best-move rewards, and compact metrics, but the wrapper defaults no longer over-weight best-move quality at the expense of overall multi-task reward.
