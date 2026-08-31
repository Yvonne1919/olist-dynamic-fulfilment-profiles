#!/usr/bin/env Rscript

# Reader-facing Chapter 4 figure assembled only from governed persisted RQ1
# outputs. This script does not fit or refit any statistical model.
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(patchwork)
  library(scales)
})

args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args_full[grep("^--file=", args_full)][1])
script_dir <- dirname(normalizePath(file_arg))
root <- normalizePath(file.path(script_dir, "../../../.."), mustWork = TRUE)
analysis_dir <- file.path(root, "analysis/rq1_speed_reliability_review_v1")
target_dir <- file.path(root, "report/thesis/images/final_ch1_ch4")
dir.create(target_dir, recursive = TRUE, showWarnings = FALSE)
v6_reader_patch <- "--v6-reader-patch" %in% commandArgs(trailingOnly = TRUE)
chapter4_patch_only <- "--chapter4-patch" %in% commandArgs(trailingOnly = TRUE)

pal <- c(ink = "#18324A", blue = "#2C6E9B", teal = "#27896F",
         orange = "#D97721", red = "#B43C4D", purple = "#73569B")

theme_thesis <- function(base_size = 8.2) {
  theme_minimal(base_size = base_size, base_family = "Helvetica") +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 1.2,
                                colour = pal[["ink"]]),
      plot.subtitle = element_text(size = base_size - 0.25, colour = "#4A5560",
                                   margin = margin(b = 4)),
      axis.title = element_text(face = "bold", size = base_size),
      axis.text = element_text(size = base_size - 0.65, colour = "#25313B"),
      legend.position = "bottom",
      legend.text = element_text(size = base_size - 0.65),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.24, colour = "#D9E0E6"),
      plot.margin = margin(4, 5, 4, 4)
    )
}

assoc <- read_csv(
  file.path(analysis_dir, "figure_sources/03_adjusted_speed_reliability_associations.csv"),
  show_col_types = FALSE
)
contrasts <- read_csv(file.path(analysis_dir, "RQ1_ADJUSTED_CONTRASTS.csv"),
                      show_col_types = FALSE)
timing <- read_csv(file.path(analysis_dir, "RQ1_REVIEW_TIMING_SENSITIVITY.csv"),
                   show_col_types = FALSE)

# Panel A: governed Model A curve and its five pre-specified percentiles.
curve <- assoc %>% filter(panel == "model_a_duration_curve")
curve_marks <- read_csv(file.path(analysis_dir, "RQ1_ADJUSTED_PROBABILITIES.csv"),
                        show_col_types = FALSE) %>%
  filter(variant == "primary", model_id == "A", estimand_type == "duration_percentile") %>%
  arrange(continuous_value) %>%
  mutate(x_numeric = continuous_value,
         point_label = paste0(toupper(sub("duration_", "", estimand_id)), "\n",
                              percent(estimate, accuracy = 0.01)),
         label_y = estimate + c(0.030, -0.030, 0.033, 0.034, 0.038))
percentile_days <- curve_marks$x_numeric

p_a <- ggplot(curve, aes(x_numeric, estimate)) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
              fill = alpha(pal[["blue"]], 0.18), colour = NA) +
  geom_line(colour = pal[["blue"]], linewidth = 0.78) +
  geom_point(data = curve_marks, colour = pal[["blue"]], size = 1.55) +
  geom_text(data = curve_marks, aes(y = label_y, label = point_label),
            size = 2.5, lineheight = 0.95, colour = pal[["ink"]]) +
  scale_x_continuous(breaks = percentile_days) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0, NA), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "A  Actual waiting and low-review probability",
       subtitle = "Month-standardised probability; HC1 95% interval",
       x = "Actual delivery duration (days)", y = "Adjusted low-review probability") +
  theme_thesis()

# Panel B: all supported late-group risk differences from Model B. The registered
# HC1 delta-Wald intervals are used consistently, including for 4--7 days late.
error_levels <- c("14+ days early", "7-13 days early", "1-6 days early",
                  "On-date", "1 day late", "2-3 days late",
                  "4-7 days late", "8+ days late")
model_b <- contrasts %>%
  filter(variant == "primary", model_id == "B",
         contrast_family == "error_group_vs_on_date") %>%
  mutate(group = recode(error_group,
                        `very early: <= -14 days` = "14+ days early",
                        `early: -13 to -7 days` = "7-13 days early",
                        `slightly early: -6 to -1 days` = "1-6 days early",
                        `on promised date` = "On-date",
                        `>=8 days late` = "8+ days late"),
         group = factor(group, levels = rev(error_levels)))

p_b <- ggplot(model_b, aes(estimate, group)) +
  geom_vline(xintercept = 0, colour = "#777777", linewidth = 0.35) +
  geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper), height = 0.16,
                 linewidth = 0.55, colour = pal[["red"]]) +
  geom_point(size = 1.85, colour = pal[["red"]]) +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     limits = c(-0.04, 0.68), breaks = c(0, 0.2, 0.4, 0.6)) +
  labs(title = "B  Promise-relative lateness and low-review risk",
       subtitle = "Risk difference versus on-date; HC1 95% intervals",
       x = "Adjusted low-review risk difference", y = NULL) +
  theme_thesis()

# Panel C: like-for-like governed full and timing-restricted contrasts. The full
# speed interval is the persisted coefficient-draw interval; the late-group
# intervals are the registered delta-Wald intervals used in Panel B and prose.
full_speed <- contrasts %>%
  filter(variant == "primary", model_id == "B", contrast_id == "C_speed") %>%
  transmute(sample = "All reviews", contrast_id,
            estimate, ci_lower, ci_upper)
full_late <- contrasts %>%
  filter(variant == "primary", model_id == "B",
         contrast_id %in% c("error_group_rd::4-7 days late",
                            "error_group_rd::>=8 days late")) %>%
  transmute(sample = "All reviews",
            contrast_id = recode(contrast_id,
              `error_group_rd::4-7 days late` = "C_late_4_7",
              `error_group_rd::>=8 days late` = "C_late_8_plus"),
            estimate, ci_lower, ci_upper)
restricted <- timing %>%
  filter(variant == "post_delivery_reviews", record_type == "contrast",
         record_id %in% c("C_speed", "C_late_4_7", "C_late_8_plus")) %>%
  transmute(sample = "At/after delivery", contrast_id = record_id,
            estimate, ci_lower, ci_upper)

timing_plot <- bind_rows(full_speed, full_late, restricted) %>%
  mutate(contrast = recode(
           contrast_id,
           C_speed = "On-date P25 to P75: 12 to 25 days",
           C_late_4_7 = "4-7 days late",
           C_late_8_plus = "8+ days late"
         ),
         contrast = factor(contrast, levels = rev(c(
           "On-date P25 to P75: 12 to 25 days", "4-7 days late", "8+ days late"
         ))))

p_c <- ggplot(timing_plot, aes(estimate, contrast, colour = sample, shape = sample)) +
  geom_vline(xintercept = 0, colour = "#777777", linewidth = 0.35) +
  geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper), height = 0.14,
                 position = position_dodge(width = 0.34), linewidth = 0.5) +
  geom_point(position = position_dodge(width = 0.34), size = 1.7) +
  scale_colour_manual(values = c("All reviews" = pal[["red"]],
                                 "At/after delivery" = pal[["blue"]])) +
  scale_x_continuous(labels = percent_format(accuracy = 1),
                     breaks = seq(0, 0.6, 0.2)) +
  labs(title = "C  Review-timing sensitivity",
       subtitle = "Full versus timing-restricted; 95% intervals",
       x = "Adjusted low-review risk difference", y = NULL,
       colour = NULL, shape = NULL) +
  theme_thesis()

figure <- p_a / (p_b | p_c) +
  plot_layout(heights = c(0.95, 1.05), widths = c(0.9, 1.1))

save_embedded <- function(plot, filename, width, height) {
  if (chapter4_patch_only && filename != "ch4_f01_model_a_curve.pdf") return(invisible(NULL))
  if (v6_reader_patch && filename == "ch3_f03b_review_adjusted.pdf") return(invisible(NULL))
  target <- file.path(target_dir, filename)
  raw <- tempfile(pattern = "rq1-traceability-", fileext = ".pdf")
  ggsave(raw, plot = plot, width = width, height = height, units = "in",
         device = function(filename, ...) grDevices::pdf(filename, useDingbats = FALSE, ...),
         bg = "white", limitsize = FALSE)
  gs <- Sys.which("gs")
  if (!nzchar(gs)) stop("Ghostscript is required for font embedding")
  status <- system2(gs, c(
    "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite",
    "-dCompatibilityLevel=1.5", "-dEmbedAllFonts=true", "-dSubsetFonts=true",
    paste0("-sOutputFile=", target), "-c",
    shQuote("<</NeverEmbed []>> setdistillerparams"), "-f", raw
  ))
  unlink(raw)
  if (!identical(status, 0L)) stop("Ghostscript failed while writing ", target)
  message("Wrote ", target)
}

save_embedded(figure, "ch3_f03b_review_adjusted.pdf", 7.15, 5.55)
save_embedded(p_a + labs(title = "Actual waiting and low-review probability"),
              "ch4_f01_model_a_curve.pdf", 6.25, 3.05)
save_embedded(p_b + labs(title = "Promise-relative lateness and low-review risk"),
              "ch4_f02_model_b_gradient.pdf", 6.25, 3.15)
# Two conceptual panels on one fixed percentage-point scale. No refitting.
timing_panels <- timing_plot %>%
  mutate(panel = if_else(contrast_id == "C_speed",
    "A  Actual waiting among on-date orders", "B  Promise-relative lateness"),
    contrast = factor(recode(contrast_id, C_speed = "12 to 25 days",
      C_late_4_7 = "4-7 days late", C_late_8_plus = "8+ days late"),
      levels = c("8+ days late", "4-7 days late", "12 to 25 days")),
    estimate_label = sprintf("%+.2f pp", 100 * estimate))
p_timing <- ggplot(timing_panels, aes(estimate * 100, contrast,
                                    colour = sample, shape = sample)) +
  geom_vline(xintercept = 0, colour = "#777777", linewidth = 0.35) +
  geom_errorbarh(aes(xmin = ci_lower * 100, xmax = ci_upper * 100),
    height = 0.14, position = position_dodge(width = 0.50), linewidth = 0.55) +
  geom_point(position = position_dodge(width = 0.50), size = 1.9) +
  geom_text(aes(x = 76, label = estimate_label),
    position = position_dodge(width = 0.50), hjust = 1, size = 2.7,
    show.legend = FALSE) +
  facet_grid(panel ~ ., scales = "free_y", space = "free_y") +
  scale_colour_manual(values = c("All reviews" = pal[["red"]],
                                 "At/after delivery" = pal[["blue"]])) +
  scale_x_continuous(limits = c(-7, 78), breaks = c(0, 20, 40, 60)) +
  labs(title = "Review timing: stable waiting contrast, attenuated lateness contrasts",
    subtitle = "Points and HC1-based 95% intervals; identical x-axis scale in both panels",
    x = "Adjusted low-review difference (percentage points)", y = NULL,
    colour = NULL, shape = NULL) +
  theme_thesis() +
  theme(strip.text.y = element_text(angle = 0, hjust = 0, face = "bold"),
        strip.placement = "outside",
        strip.background = element_rect(fill = "#EEF3F8", colour = NA)) +
  theme(strip.text.y.right = element_text(angle = 0))
# Put facet titles above their panels rather than in a wide right-side strip.
p_timing <- p_timing + facet_wrap(~panel, ncol = 1, scales = "free_y") +
  theme(strip.text = element_text(face = "bold", hjust = 0),
        strip.text.y = element_blank())
save_embedded(p_timing, "ch4_f03_review_timing.pdf", 7.15, 3.70)

message("All figures use persisted governed RQ1 tables; no model was fitted.")
