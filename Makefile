
RESULTS_DIR := __results__

# Run models

all: \
	qlearning \
	policy \

qlearning: \
	qlearning_all.png \
	qlearning_avg.png \

policy: \
	policy_all.png \
	policy_batch\:32_all.png \
	policy_batch\:512_all.png \
	policy_discount\:0.5_all.png \
	policy_discount\:0.99_all.png \

# Global

.SECONDARY:

%_one.png: $(RESULTS_DIR)/%_run1/results.csv
	@echo ./train.py plot '->' $@
	@PLOT_FILE=$@ ./train.py plot $^

%_all.png: \
		$(RESULTS_DIR)/%_run1/results.csv \
		$(RESULTS_DIR)/%_run2/results.csv \
		$(RESULTS_DIR)/%_run3/results.csv \
		$(RESULTS_DIR)/%_run4/results.csv
	@echo ./train.py plot '->' $@
	@PLOT_FILE=$@ ./train.py plot $^

%_avg.png: $(RESULTS_DIR)/avg/%_avg4
	@echo ./train.py plot '->' $@
	@PLOT_FILE=$@ ./train.py plot $^

$(RESULTS_DIR)/%/results.csv:
	@echo ./$(firstword $(subst _, ,$*)).py '->' $@
	@LOG_DIR=$(RESULTS_DIR)/$* \
		./$(firstword $(subst _, ,$*)).py \
			$(subst $(firstword $(subst _, ,$*))_,,$*)

$(RESULTS_DIR)/avg/%_avg4: \
		$(RESULTS_DIR)/%_run1/results.csv \
		$(RESULTS_DIR)/%_run2/results.csv \
		$(RESULTS_DIR)/%_run3/results.csv \
		$(RESULTS_DIR)/%_run4/results.csv
	@mkdir -p $(dir $@)
	@cat $^ | sort -n | tail -n +4 > $@
