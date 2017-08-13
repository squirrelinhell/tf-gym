
R := __results__

# Run models

all: \
	qlearning \
	policy \

qlearning: \
	$R/agent\:qlearning_env\:FrozenLake-v0_4.png

policy: \
	$R/agent\:policy_env\:CartPole-v1_normalize-adv\:0.5_16.png \
	$R/agent\:policy_env\:CartPole-v1_normalize-obs\:0.00002_16.png \
	$R/agent\:policy_env\:CartPole-v1_normalize-obs\:0.0001_16.png \
	$R/agent\:policy_env\:CartPole-v1_batch\:10_lr\:0.002_16.png \
	$R/agent\:policy_env\:CartPole-v1_batch\:100_lr\:0.02_16.png \
	$R/agent\:policy_env\:CartPole-v1_batch\:1000_lr\:0.2_16.png \

# Global

.SECONDARY:

$R/%_1.png: $R/run/%/results.csv
	@echo ./plot.py '->' $@
	@PLOT_FILE=$@ ./plot.py $^

$R/%_4.png: \
		$R/run/%_run1/results.csv \
		$R/run/%_run2/results.csv \
		$R/run/%_run3/results.csv \
		$R/run/%_run4/results.csv
	@echo ./plot.py '->' $@
	@PLOT_FILE=$@ ./plot.py $^

$R/%_16.png: \
		$R/avg/%_batch1_avg \
		$R/avg/%_batch2_avg \
		$R/avg/%_batch3_avg \
		$R/avg/%_batch4_avg
	@echo ./plot.py '->' $@
	@PLOT_FILE=$@ ./plot.py $^

$R/run/%/results.csv:
	@echo ./train.py '->' $@
	@LOG_DIR=$R/run/$* ./train.py $*

$R/avg/%_avg: \
		$R/run/%_run1/results.csv \
		$R/run/%_run2/results.csv \
		$R/run/%_run3/results.csv \
		$R/run/%_run4/results.csv
	@mkdir -p $(dir $@)
	@cat $^ | sort -n | tail -n +4 > $@
