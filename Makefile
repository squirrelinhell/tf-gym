
R := __results__
COMMA := ,

# Run models

all: \
	qlearning \
	policy_gradient \
	evolution_strategy \

qlearning: \
	$R/qlearning.png

policy_gradient: \
	$R/policy_gradient,end_reward\:-100.png \
	$R/policy_gradient,end_reward\:-100,batch\:512,lr\:0.03.png \
	$R/policy_gradient,end_reward\:-100,normalize_obs\:0.00003.png \
	$R/policy_gradient,end_reward\:-100,normalize_adv\:0.5.png \

evolution_strategy: \
	$R/evolution_strategy.png \

# Global

.SECONDARY:

$R/%,1.png: \
		$R/log/%/results.csv
	@echo ./utils/plot.py '->' $@
	@PLOT_FILE=$@ ./utils/plot.py $^

$R/%.png: \
		$R/log/%,run1/results.csv \
		$R/log/%,run2/results.csv \
		$R/log/%,run3/results.csv \
		$R/log/%,run4/results.csv
	@echo ./utils/plot.py '->' $@
	@PLOT_FILE=$@ ./utils/plot.py $^

$R/log/%/results.csv:
	@echo ./$(firstword $(subst $(COMMA), ,$*)).py '->' $@
	@mkdir -p $(dir $@)
	@LOG_DIR=$(dir $@) ./$(firstword $(subst $(COMMA), ,$*)).py \
		$(subst $(firstword $(subst $(COMMA), ,$*))$(COMMA),,$*)
	@mv $(dir $@)/episodes.csv $@
