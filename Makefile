
help:
	@echo
	@echo "Run models:"
	@echo "   make qlearning"
	@echo "   make policy"
	@echo

RESULTS_DIR := __results__

# Run models

qlearning: $(RESULTS_DIR)/qlearning_run5.png

$(RESULTS_DIR)/qlearning_%.csv: qlearning.py
	@mkdir -p $(dir $@)
	RESULTS_FILE=$@ ./qlearning.py $*

policy: $(RESULTS_DIR)/policy_run5.png

$(RESULTS_DIR)/policy_%.csv: policy.py
	@mkdir -p $(dir $@)
	RESULTS_FILE=$@ ./policy.py $*

# Global

.SECONDARY:

$(RESULTS_DIR)/%_run5.png: \
		$(RESULTS_DIR)/%_run1.csv \
		$(RESULTS_DIR)/%_run2.csv \
		$(RESULTS_DIR)/%_run3.csv \
		$(RESULTS_DIR)/%_run4.csv \
		$(RESULTS_DIR)/%_run5.csv
	PLOT_FILE=$@ ./train.py plot $^
