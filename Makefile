
R := __results__
COMMA := ,

# Run models

all: \
	qlearning \
	policy \

qlearning: \
	$R/qlearning.png

policy: \
	$R/policy.png \
	$R/policy,batch\:512,lr\:0.03.png \
	$R/policy,normalize_obs\:0.00003.png \
	$R/policy,normalize_adv\:0.5.png \

# Global

.SECONDARY:

$R/%,1.png: \
		$R/run/%/results.csv
	@echo ./lib/plot.py '->' $@
	@PLOT_FILE=$@ ./lib/plot.py $^

$R/%.png: \
		$R/run/%,run1/results.csv \
		$R/run/%,run2/results.csv \
		$R/run/%,run3/results.csv \
		$R/run/%,run4/results.csv
	@echo ./lib/plot.py '->' $@
	@PLOT_FILE=$@ ./lib/plot.py $^

$R/run/%/results.csv:
	@echo ./$(firstword $(subst $(COMMA), ,$*)).py '->' $@
	@LOG_DIR=$R/run/$* ./$(firstword $(subst $(COMMA), ,$*)).py \
		$(subst $(firstword $(subst $(COMMA), ,$*))$(COMMA),,$*)
