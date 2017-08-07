
R := __results__

# Run models

all: \
	qlearning \
	policy \

qlearning: \
	$R/qlearning_show4.png \
	$R/qlearning_avg4.png \

policy: \
	$R/policy_compare_batch.png \

$R/policy_compare_batch.png: \
	$R/policy_batch\:100_lr\:0.02_avg16.png \
	$R/policy_batch\:10_lr\:0.02_avg16.png \
	$R/policy_batch\:10_lr\:0.002_avg16.png \
	$R/policy_batch\:1000_lr\:0.02_avg16.png \
	$R/policy_batch\:1000_lr\:0.2_avg16.png \

	@echo ./train.py plot '->' $@
	@PLOT_FILE=$@ ./train.py plot $(patsubst $R/%.png,$R/avg/%,$^)

# Global

.SECONDARY:

$R/%.png: $R/run/%_run1/results.csv
	@echo ./train.py plot '->' $@
	@PLOT_FILE=$@ ./train.py plot $^

$R/%_show4.png: \
		$R/run/%_run1/results.csv \
		$R/run/%_run2/results.csv \
		$R/run/%_run3/results.csv \
		$R/run/%_run4/results.csv
	@echo ./train.py plot '->' $@
	@PLOT_FILE=$@ ./train.py plot $^

$R/%_avg4.png: $R/avg/%_avg4
	@echo ./train.py plot '->' $@
	@PLOT_FILE=$@ ./train.py plot $^

$R/%_avg16.png: $R/avg/%_avg16
	@echo ./train.py plot '->' $@
	@PLOT_FILE=$@ ./train.py plot $^

$R/run/%/results.csv:
	@echo ./$(firstword $(subst _, ,$*)).py '->' $@
	@LOG_DIR=$R/run/$* \
		./$(firstword $(subst _, ,$*)).py \
			$(subst $(firstword $(subst _, ,$*))_,,$*)

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
