
R := __results__

# Run models

all: \
	qlearning \
	policy \

qlearning: \
	$R/env\:FrozenLake-v0_agent\:qlearning_show4.png

policy: \
	$R/policy_normalize.png \
	$R/policy_batch.png \

$R/policy_normalize.png: \
	$R/avg/env\:CartPole-v1_agent\:policy_normalize\:mean_avg16 \
	$R/avg/env\:CartPole-v1_agent\:policy_normalize\:off_avg16 \

	@echo ./plot.py '->' $@
	@PLOT_FILE=$@ ./plot.py $(patsubst $R/%.png,$R/avg/%,$^)

$R/policy_batch.png: \
	$R/avg/env\:CartPole-v1_agent\:policy_batch\:100_lr\:0.02_avg16 \
	$R/avg/env\:CartPole-v1_agent\:policy_batch\:10_lr\:0.02_avg16 \
	$R/avg/env\:CartPole-v1_agent\:policy_batch\:10_lr\:0.002_avg16 \
	$R/avg/env\:CartPole-v1_agent\:policy_batch\:1000_lr\:0.02_avg16 \
	$R/avg/env\:CartPole-v1_agent\:policy_batch\:1000_lr\:0.2_avg16 \

	@echo ./plot.py '->' $@
	@PLOT_FILE=$@ ./plot.py $(patsubst $R/%.png,$R/avg/%,$^)

# Global

.SECONDARY:

$R/%.png: $R/run/%_run1/results.csv
	@echo ./plot.py '->' $@
	@PLOT_FILE=$@ ./plot.py $^

$R/%_show4.png: \
		$R/run/%_run1/results.csv \
		$R/run/%_run2/results.csv \
		$R/run/%_run3/results.csv \
		$R/run/%_run4/results.csv
	@echo ./plot.py '->' $@
	@PLOT_FILE=$@ ./plot.py $^

$R/%_avg4.png: $R/avg/%_avg4
	@echo ./plot.py '->' $@
	@PLOT_FILE=$@ ./plot.py $^

$R/%_avg16.png: $R/avg/%_avg16
	@echo ./plot.py '->' $@
	@PLOT_FILE=$@ ./plot.py $^

$R/run/%/results.csv:
	@echo ./train.py '->' $@
	@LOG_DIR=$R/run/$* ./train.py $*

$R/avg/%_avg4: \
		$R/run/%_run1/results.csv \
		$R/run/%_run2/results.csv \
		$R/run/%_run3/results.csv \
		$R/run/%_run4/results.csv
	@mkdir -p $(dir $@)
	@cat $^ | sort -n | tail -n +4 > $@

$R/avg/%_avg16: \
		$R/avg/%_run1_avg4 \
		$R/avg/%_run2_avg4 \
		$R/avg/%_run3_avg4 \
		$R/avg/%_run4_avg4
	@mkdir -p $(dir $@)
	@cat $^ | sort -n | tail -n +4 > $@
