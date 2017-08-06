
RESULTS_DIR := __results__

# Run models

all: \
	qlearning \
	policy \

qlearning: \
	$(RESULTS_DIR)/qlearning_avg.png \
	$(RESULTS_DIR)/qlearning_all.png \

policy: \
	$(RESULTS_DIR)/policy_batch.png \
	$(RESULTS_DIR)/policy_discount.png \
	$(RESULTS_DIR)/policy_stepreward.png \

$(RESULTS_DIR)/policy_batch.png: \
	$(RESULTS_DIR)/avg/policy.csv \
	$(RESULTS_DIR)/avg/policy_batch32.csv \
	$(RESULTS_DIR)/avg/policy_batch512.csv
	@echo ./train.py plot '->' $@
	@PLOT_FILE=$@ ./train.py plot $^

$(RESULTS_DIR)/policy_discount.png: \
	$(RESULTS_DIR)/avg/policy.csv \
	$(RESULTS_DIR)/avg/policy_discount0.7.csv \
	$(RESULTS_DIR)/avg/policy_discount0.99.csv
	@echo ./train.py plot '->' $@
	@PLOT_FILE=$@ ./train.py plot $^

$(RESULTS_DIR)/policy_stepreward.png: \
	$(RESULTS_DIR)/avg/policy.csv \
	$(RESULTS_DIR)/avg/policy_stepreward0.csv \
	$(RESULTS_DIR)/avg/policy_stepreward0.1.csv
	@echo ./train.py plot '->' $@
	@PLOT_FILE=$@ ./train.py plot $^

# Global

.SECONDARY:

$(RESULTS_DIR)/%_all.png: \
		$(RESULTS_DIR)/run/%_run1.csv \
		$(RESULTS_DIR)/run/%_run2.csv \
		$(RESULTS_DIR)/run/%_run3.csv \
		$(RESULTS_DIR)/run/%_run4.csv
	@echo ./train.py plot '->' $@
	@PLOT_FILE=$@ ./train.py plot $^

$(RESULTS_DIR)/%_avg.png: $(RESULTS_DIR)/avg/%.csv
	@echo ./train.py plot '->' $@
	@PLOT_FILE=$@ ./train.py plot $^

$(RESULTS_DIR)/run/%.csv:
	@echo ./$(firstword $(subst _, ,$*)).py '->' $@
	@RESULTS_FILE=$@ ./$(firstword $(subst _, ,$*)).py \
		$(subst $(firstword $(subst _, ,$*))_,,$*)

$(RESULTS_DIR)/avg/%.csv: \
		$(RESULTS_DIR)/run/%_run1.csv \
		$(RESULTS_DIR)/run/%_run2.csv \
		$(RESULTS_DIR)/run/%_run3.csv \
		$(RESULTS_DIR)/run/%_run4.csv
	@mkdir -p $(dir $@)
	@cat $^ | sort -n | tail -n +4 > $@
