
RESULTS_DIR := __results__

# Run models

all: \
	qlearning \
	policy \

qlearning: \
	$(RESULTS_DIR)/qlearning_run5.png \

$(RESULTS_DIR)/qlearning_%.csv: qlearning.py
	RESULTS_FILE=$@ ./qlearning.py $*

policy: \
	$(RESULTS_DIR)/policy_decay5_run5.png \
	$(RESULTS_DIR)/policy_decay10_run5.png \

$(RESULTS_DIR)/policy_%.csv: policy.py
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
