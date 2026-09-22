#!/usr/bin/env Rscript
# Simple GLMM framework to find predictors of GFP intensity from other channel metrics
# Usage: Rscript glmm.R data.csv [response] [predictor_prefix] [random_column]

csv <- "D:/LNP/Mich2data/08-09-2026_Huh7&HepG2 nanowells w GFP/100 nm/Processed Wells/C01/results_puncta/Merged_mCherry_GFP_results.csv"

pred_prefix <- c("puncta")
random_col <- ifelse(length(args) >= 4, args[4], NA)

library(lme4)
library(glmmTMB)
library(car)
#library(readxl)

dat <- read.csv(csv) # read_excel(csv) for xlsx filepath

# select predictors by prefix (numeric)
preds <- names(dat)[startsWith(names(dat), pred_prefix)]
if(length(preds) == 0) stop("No predictors found with provided prefix")

# Specify predictor
response <- c("GFP_I_by_area") 
    #"GFP_I_mean"
    #"GFP_I_sum"
    #"GFP_I_by_area"
    #"GFP_I_max"
    #"GFP_I_sd"

# keep only numeric predictors and response
keep <- c(response, preds, random_col)
keep <- keep[!is.na(keep)]
dat2 <- dat[ , intersect(names(dat), keep)]
dat2 <- na.omit(dat2)

# scale predictors
dat2[ preds ] <- scale(dat2[ preds ])

# build formula
fixed_formula <- paste(response, "~", paste(preds, collapse = " + "))
if(!is.na(random_col) && random_col %in% names(dat2)){
	form <- as.formula(paste(fixed_formula, "+ (1|", random_col, ")"))
} else {
	form <- as.formula(fixed_formula)
}

cat("Formula:", deparse(form), "\n")

# fit linear mixed model (Gaussian) and a gamma model if response positive
fit_lmm <- tryCatch({
	lmer(form, data = dat2)
}, error = function(e) NULL)

fit_tmb <- NULL
if(all(dat2[[response]] > 0)){
	fit_tmb <- tryCatch({
		glmmTMB(formula = form, family = Gamma(link = "log"), data = dat2)
	}, error = function(e) NULL)
}

cat("Observations:", nrow(dat2), "\n")
if(!is.null(fit_lmm)){
	cat("--- lmer summary ---\n")
	print(summary(fit_lmm))
	cat("ANOVA:\n")
	print(anova(fit_lmm))
	cat("VIF (fixed effects):\n")
	if(length(preds) > 1) print(vif(lm(as.formula(paste(response, "~", paste(preds, collapse = "+"))), data = dat2)))
}

if(!is.null(fit_tmb)){
	cat("--- glmmTMB (Gamma) summary ---\n")
	print(summary(fit_tmb))
}

cat("Done.\n")
