
R := __results__

# Run models

all: \
	qlearning \
	policy \

qlearning: \
	$R/qlearning_env\:FrozenLake-v0_steps\:20000.png

policy: \
	$R/policy_env\:CartPole-v1.png \
	$R/policy_env\:CartPole-v1_batch\:512_lr\:0.03.png \
	$R/policy_env\:CartPole-v1_normalize-obs\:0.00003.png \
	$R/policy_env\:CartPole-v1_normalize-adv\:0.5.png \
	$R/policy_env\:CartPole-v1_value-grad\:0.1.png \
	$R/policy_env\:CartPole-v1_value-grad\:0.01.png \
	$R/policy_env\:CartPole-v1_value-grad\:0.01_normalize-adv\:0.5.png \

# Global

.SECONDARY:

$R/%_run1.png: \
		$R/run/%_run1/results.csv
	@echo ./plot.py '->' $@
	@PLOT_FILE=$@ ./plot.py $^

$R/%.png: \
		$R/run/%_run1/results.csv \
		$R/run/%_run2/results.csv \
		$R/run/%_run3/results.csv \
		$R/run/%_run4/results.csv \
		$R/run/%_run5/results.csv
	@echo ./plot.py '->' $@
	@PLOT_FILE=$@ ./plot.py $^

$R/run/%/results.csv:
	@echo ./train.py '->' $@
	@LOG_DIR=$R/run/$* ./train.py agent\:$*
