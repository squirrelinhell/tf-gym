
R := __results__

# Run models

all: \
	qlearning \
	policy \

qlearning: \
	$R/agent\:qlearning_env\:FrozenLake-v0_3.png

policy: \
	$R/agent\:policy_env\:CartPole-v1_normalize-adv\:0.5_6.png \
	$R/agent\:policy_env\:CartPole-v1_normalize-obs\:0.00002_6.png \
	$R/agent\:policy_env\:CartPole-v1_normalize-obs\:0.0001_6.png \
	$R/agent\:policy_env\:CartPole-v1_batch\:10_lr\:0.002_6.png \
	$R/agent\:policy_env\:CartPole-v1_batch\:100_lr\:0.02_6.png \
	$R/agent\:policy_env\:CartPole-v1_batch\:1000_lr\:0.2_6.png \

# Global

.SECONDARY:

$R/%_1.png: \
		$R/run/%_run1/results.csv
	@echo ./plot.py '->' $@
	@PLOT_FILE=$@ ./plot.py $^

$R/%_3.png: \
		$R/run/%_run1/results.csv \
		$R/run/%_run2/results.csv \
		$R/run/%_run3/results.csv
	@echo ./plot.py '->' $@
	@PLOT_FILE=$@ ./plot.py $^

$R/%_6.png: \
		$R/run/%_run1/results.csv \
		$R/run/%_run2/results.csv \
		$R/run/%_run3/results.csv \
		$R/run/%_run4/results.csv \
		$R/run/%_run5/results.csv \
		$R/run/%_run6/results.csv
	@echo ./plot.py '->' $@
	@PLOT_FILE=$@ ./plot.py $^

$R/run/%/results.csv:
	@echo ./train.py '->' $@
	@LOG_DIR=$R/run/$* ./train.py $*
