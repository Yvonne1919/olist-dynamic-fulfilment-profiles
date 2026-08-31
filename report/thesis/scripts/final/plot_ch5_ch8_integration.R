#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(stringr)
})

command_args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", command_args[grep("^--file=", command_args)][1])
script_dir <- dirname(normalizePath(file_arg))
root <- normalizePath(file.path(script_dir, "../../../.."), mustWork = TRUE)
chapter5_patch_only <- "--chapter5-patch" %in% commandArgs(trailingOnly = TRUE)
final_reader_patch <- "--final-reader-patch" %in% commandArgs(trailingOnly = TRUE)
reader_polish <- "--reader-polish" %in% commandArgs(trailingOnly = TRUE)
v6_reader_patch <- "--v6-reader-patch" %in% commandArgs(trailingOnly = TRUE)
v4_probability_only <- "--v4-probability-only" %in% commandArgs(trailingOnly = TRUE)

fig_dir <- file.path(root, "report/thesis/images/final")
src_dir <- file.path(root, "report/thesis/figure_sources/final")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(src_dir, recursive = TRUE, showWarnings = FALSE)

pal <- c(
  seller = "#D55E00", state_od = "#0072B2", both = "#009E73",
  dark = "#23395D", amber = "#E69F00", adverse = "#B23A48",
  favourable = "#087E5B", pale = "#EEF3F8", mid = "#7B8794"
)

theme_thesis <- function(base_size = 8.7) {
  theme_minimal(base_size = base_size, base_family = "Helvetica") +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 1.4, colour = pal[["dark"]]),
      plot.subtitle = element_text(size = base_size - 0.1, colour = "#444444", margin = margin(b = 5)),
      axis.title = element_text(face = "bold", size = base_size),
      axis.text = element_text(size = base_size - 0.5, colour = "#222222"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.25, colour = "#D9DEE5"),
      strip.text = element_text(face = "bold", colour = pal[["dark"]]),
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      legend.key.height = grid::unit(3.5, "mm"),
      plot.margin = margin(5, 7, 5, 5)
    )
}

save_pdf <- function(plot, name, width = 7.15, height = 4.35) {
  if (v6_reader_patch && !name %in% c("f04a_profile_selection_funnel.pdf", "f05_profile_stability_tiers.pdf")) return(invisible(NULL))
  if (v4_probability_only && name != "f06_direct_breach_model_families.pdf") return(invisible(NULL))
  if (reader_polish && !name %in% c("f06_direct_breach_model_families.pdf",
                                   "f09_terminal_model_family_transfer.pdf")) return(invisible(NULL))
  if (final_reader_patch && !name %in% c("f07_direct_promise_reliability.pdf",
    "f08a_conditional_severity_skill.pdf", "f08b_conditional_q90_coverage.pdf")) return(invisible(NULL))
  if (chapter5_patch_only && !name %in% c(
    "f04a_profile_selection_funnel.pdf", "f04b_raw_eb_later_loss.pdf",
    "f05a_history_support_smoothness.pdf", "f05b_history_later_process_loss.pdf"
  )) return(invisible(NULL))
  filename <- file.path(fig_dir, name)
  raw_pdf <- tempfile(pattern = "ch5-ch8-", fileext = ".pdf")
  ggsave(
    raw_pdf, plot = plot, width = width, height = height, units = "in",
    device = function(filename, ...) grDevices::pdf(filename, useDingbats = FALSE, ...),
    bg = "white", limitsize = FALSE
  )
  gs_bin <- Sys.which("gs")
  if (!nzchar(gs_bin)) stop("Ghostscript is required to embed figure fonts")
  status <- system2(
    gs_bin,
    c(
      "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite",
      "-dCompatibilityLevel=1.5", "-dEmbedAllFonts=true", "-dSubsetFonts=true",
      paste0("-sOutputFile=", filename), "-c",
      shQuote("<</NeverEmbed []>> setdistillerparams"), "-f", raw_pdf
    )
  )
  unlink(raw_pdf)
  if (!identical(status, 0L)) stop("Ghostscript font embedding failed for ", name)
}

write_source <- function(data, name) {
  if (v6_reader_patch || chapter5_patch_only || final_reader_patch || reader_polish || v4_probability_only) return(invisible(NULL))
  write_csv(data, file.path(src_dir, name), na = "")
}

block_labels <- c(seller = "Seller", state_od = "State-OD", both = "Both")
block_colours <- c(Seller = pal[["seller"]], `State-OD` = pal[["state_od"]], Both = pal[["both"]])

# -----------------------------------------------------------------------------
# Figure 5.1: candidate funnel and reader-facing empirical-Bayes comparison
# -----------------------------------------------------------------------------
funnel <- read_csv(
  file.path(src_dir, "F4_profile_selection_funnel.csv"),
  show_col_types = FALSE
) %>%
  mutate(
    stage = recode(
      stage,
      `Candidate rules` = "Profile designs",
      `Evidence eligible` = "Usable history",
      `Non-dominated` = "Best trade-offs",
      Selected = "Later-process test",
      Eligible = "Ready for\norder study",
      Representatives = "Profiles used\nin models"
    ),
    detail = recode(
      detail,
      `Full development grid` = "All combinations",
      `Minimum evidence passes` = "Enough history and later rows",
      `Pareto frontier` = "No better design on every criterion",
      `Subsequent confirmation` = "Assessed in later months",
      `2 Strong + 12 Partial` = "Meet order-study requirements",
      `4 process + 1 secondary endpoint` = "Four process + one secondary"
    )
  )
shrinkage <- read_csv(
  file.path(src_dir, "F4C_matched_selected_P1_shrinkage.csv"),
  show_col_types = FALSE
)

phase_counts <- funnel %>% arrange(stage_order) %>% pull(count)
stopifnot(length(phase_counts) == 6)
phase_data <- tibble(
  x = 1:3,
  phase = c("Feasibility", "Development selection", "Future validation"),
  counts = c(paste(comma(phase_counts[1:2]), collapse = "  >  "),
             paste(comma(phase_counts[2:4]), collapse = "  >  "),
             paste(comma(phase_counts[4:6]), collapse = "  >  ")),
  detail = c("Profile designs > usable history",
    "Usable history > Pareto trade-offs\n> later-process finalists",
    "Finalists > meet process/support rules\n> representatives"))
p_funnel <- ggplot(phase_data, aes(x, 1)) +
  geom_rect(aes(xmin = x - .46, xmax = x + .46, ymin = .15, ymax = 1.25),
    fill = "#EEF3F8", colour = "#AAB8C6", linewidth = .4) +
  geom_text(aes(y = 1.07, label = phase), fontface = "bold", size = 3.1,
    colour = pal[["dark"]]) +
  geom_text(aes(y = .78, label = counts), fontface = "bold", size = 3.55,
    colour = pal[["dark"]]) +
  geom_text(aes(y = .41, label = detail), size = 2.6, lineheight = 1.05,
    colour = "#4B5563") +
  geom_segment(data = filter(phase_data, x < 3),
    aes(x = x + .47, xend = x + .53, y = .78, yend = .78),
    arrow = arrow(length = grid::unit(1.5, "mm"), type = "closed"),
    colour = pal[["dark"]]) +
  coord_cartesian(xlim = c(.5, 3.5), ylim = c(.12, 1.3), clip = "off") +
  theme_void(base_family = "Helvetica") +
  theme(plot.title = element_text(face = "bold", size = 10, colour = pal[["dark"]]),
        plot.margin = margin(4, 5, 3, 5))

shrink_long <- shrinkage %>%
  pivot_longer(c(raw, shrinkage), names_to = "estimator", values_to = "loss")
p_shrink <- ggplot(shrink_long, aes(x = loss, y = metric, colour = estimator, shape = estimator)) +
  geom_line(aes(group = metric), colour = "#BFC7D1", linewidth = 1.0) +
  geom_point(size = 2.0) +
  scale_colour_manual(
    values = c(raw = "#777777", shrinkage = pal[["seller"]]),
    labels = c(raw = "Raw", shrinkage = "Empirical Bayes")
  ) +
  scale_shape_manual(
    values = c(raw = 16, shrinkage = 17),
    labels = c(raw = "Raw", shrinkage = "Empirical Bayes")
  ) +
  facet_wrap(~metric, scales = "free_x", ncol = 2) +
  labs(
    title = "B  How shrinkage changes later prediction loss",
    subtitle = "Identical later rows; lower loss is better",
    x = "Median future-row loss", y = NULL, colour = NULL, shape = NULL
  ) +
  theme_thesis(8.3) +
  theme(strip.background = element_blank(), axis.text.y = element_blank())

p_selection <- p_funnel / p_shrink + plot_layout(heights = c(0.72, 1.45))
save_pdf(p_selection, "f04_profile_selection_shrinkage.pdf", height = 4.35)
save_pdf(p_funnel + labs(title = "How are profile definitions narrowed down?"),
         "f04a_profile_selection_funnel.pdf", height = 1.85)
save_pdf(p_shrink + labs(title = "How does shrinkage change later loss?"),
         "f04b_raw_eb_later_loss.pdf", height = 2.80)

# -----------------------------------------------------------------------------
# Figure 5.2: bounded recent history versus all mature history
# -----------------------------------------------------------------------------
am_root <- file.path(root, "analysis/all_mature_history_sensitivity_v1")

support_raw <- read_csv(
  file.path(am_root, "SUPPORT_COVERAGE_COLDSTART_COMPARISON.csv"),
  show_col_types = FALSE
)

support_summary <- support_raw %>%
  filter(
    period == "confirmation", horizon_days == 7,
    population == "future_orders",
    metric %in% c("support_median", "cold_start_share_all_placed")
  ) %>%
  group_by(profile_code, metric) %>%
  summarise(
    selected_90d_value = median(selected_90d_value, na.rm = TRUE),
    all_mature_value = median(all_mature_value, na.rm = TRUE),
    .groups = "drop"
  )

uncertainty_raw <- read_csv(
  file.path(am_root, "UNCERTAINTY_STABILITY_COMPARISON.csv"),
  show_col_types = FALSE
)

uncertainty_summary <- uncertainty_raw %>%
  filter(
    period == "confirmation", horizon_days == 7,
    population == "future_seen_orders", support_stratum == "all_support",
    metric == "posterior_se_median"
  ) %>%
  group_by(profile_code, metric) %>%
  summarise(
    selected_90d_value = median(selected_90d_value, na.rm = TRUE),
    all_mature_value = median(all_mature_value, na.rm = TRUE),
    .groups = "drop"
  )

tier_summary <- uncertainty_raw %>%
  filter(
    period == "confirmation", is.na(horizon_days),
    component == "daily_stability",
    metric == "pct_entities_changing_level"
  ) %>%
  group_by(profile_code, metric) %>%
  summarise(
    selected_90d_value = median(selected_90d_value, na.rm = TRUE),
    all_mature_value = median(all_mature_value, na.rm = TRUE),
    .groups = "drop"
  )

support_matrix <- bind_rows(support_summary, uncertainty_summary, tier_summary) %>%
  mutate(
    metric_label = recode(
      metric,
      support_median = "Median support",
      posterior_se_median = "Score uncertainty",
      cold_start_share_all_placed = "Cold-start share",
      pct_entities_changing_level = "Daily tier change"
    ),
    display = case_when(
      metric == "support_median" ~ paste0(comma(round(selected_90d_value)), " /\n", comma(round(all_mature_value))),
      metric == "posterior_se_median" ~ paste0(sprintf("%.3f", selected_90d_value), " /\n", sprintf("%.3f", all_mature_value)),
      TRUE ~ paste0(percent(selected_90d_value, accuracy = 0.01), " /\n", percent(all_mature_value, accuracy = 0.01))
    ),
    profile_label = recode(
      profile_code,
      S1 = "Seller level",
      S2 = "Seller tail risk",
      R1 = "State-OD level",
      R2 = "State-OD tail risk"
    ),
    profile_label = factor(
      profile_label,
      levels = rev(c("Seller level", "Seller tail risk", "State-OD level", "State-OD tail risk"))
    ),
    metric_label = factor(
      metric_label,
      levels = c("Median support", "Score uncertainty", "Cold-start share", "Daily tier change"),
      labels = c("Median\nsupport", "Score\nuncertainty", "Cold-start\nshare", "Daily tier\nchange")
    )
  )
write_source(support_matrix, "F5A_all_mature_support_stability.csv")

p_f5a <- ggplot(support_matrix, aes(x = metric_label, y = profile_label)) +
  geom_tile(fill = "#EEF4F8", colour = "white", linewidth = 1.0) +
  geom_text(aes(label = display), size = 3.0, lineheight = 0.9, colour = pal[["dark"]]) +
  labs(
    title = "A  Support and smoothness",
    subtitle = "Cells show 90-day / cumulative history; Jan-Jun later outcomes",
    x = NULL, y = NULL
  ) +
  theme_thesis(8.5) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(face = "bold", size = 7.6),
    axis.text.y = element_text(face = "bold"),
    legend.position = "none"
  )

loss_summary <- read_csv(
  file.path(am_root, "STANDALONE_90D_VS_ALL_MATURE_SUMMARY.csv"),
  show_col_types = FALSE
) %>%
  filter(
    period %in% c("development", "confirmation"), horizon_days == 7,
    (profile_code %in% c("S1", "R1") & metric == "log_mae") |
      (profile_code %in% c("S2", "R2") & metric == "log_loss")
  ) %>%
  mutate(
    delta_x1000 = 1000 * all_mature_minus_90d,
    direction = if_else(all_mature_minus_90d < 0, "Cumulative history better", "90-day better"),
    period_label = recode(period, development = "Development", confirmation = "Jan-Jun later outcomes"),
    profile_label = recode(
      profile_code,
      S1 = "Seller level",
      S2 = "Seller tail risk",
      R1 = "State-OD level",
      R2 = "State-OD tail risk"
    ),
    profile_label = factor(
      profile_label,
      levels = rev(c("Seller level", "Seller tail risk", "State-OD level", "State-OD tail risk"))
    ),
    value_label = sprintf("%+.2f", delta_x1000)
  )
write_source(loss_summary, "F5B_all_mature_primary_loss.csv")

p_f5b <- ggplot(loss_summary, aes(x = delta_x1000, y = profile_label, colour = direction)) +
  geom_vline(xintercept = 0, linewidth = 0.45, colour = "#555555") +
  geom_segment(aes(x = 0, xend = delta_x1000, yend = profile_label), linewidth = 0.7) +
  geom_point(size = 2.2) +
  geom_text(aes(label = value_label), nudge_y = 0.22, size = 2.45, show.legend = FALSE) +
  facet_wrap(~period_label, ncol = 1) +
  scale_colour_manual(values = c(
    `Cumulative history better` = pal[["favourable"]],
    `90-day better` = pal[["adverse"]]
  )) +
  scale_x_continuous(expand = expansion(mult = c(0.12, 0.16))) +
  labs(
    title = "B  Later-process prediction",
    subtitle = "1 of 8 comparisons favours cumulative history",
    x = "Later-loss difference x 1000\n(negative favours cumulative history)", y = NULL,
    colour = NULL
  ) +
  theme_thesis(8.5) +
  theme(legend.position = "bottom", legend.text = element_text(size = 6.7))

p_f5 <- p_f5a | p_f5b
p_f5 <- p_f5 + plot_layout(widths = c(1.3, 1.1))
save_pdf(p_f5, "f05_all_mature_history.pdf", height = 4.25)
save_pdf(p_f5a + labs(title = "Does cumulative history make scores smoother?"),
         "f05a_history_support_smoothness.pdf", height = 2.80)
save_pdf(p_f5b + labs(title = "Does cumulative history track later outcomes better?") +
           theme(legend.position = "none"),
         "f05b_history_later_process_loss.pdf", width = 6.25, height = 3.65)

if (chapter5_patch_only) {
  message("Wrote four split Chapter 5 presentation PDFs only; persisted numerical sources unchanged; no model fitted.")
  quit(save = "no", status = 0)
}

# -----------------------------------------------------------------------------
# Figure 5.3: standalone future-process transfer and communication tiers
# -----------------------------------------------------------------------------
f5_transfer <- read_csv(
  file.path(src_dir, "F5_standalone_confirmation.csv"),
  show_col_types = FALSE
) %>%
  mutate(
    metric_family = recode(
      metric_family,
      "Binary: reference - candidate log loss" = "Binary: log-loss improvement",
      "Continuous: parent - candidate log-MAE" = "Continuous: log-MAE improvement"
    ),
    process = factor(process, levels = rev(process))
  )

f5_transfer_long <- f5_transfer %>%
  pivot_longer(c(development, confirmation), names_to = "period", values_to = "improvement") %>%
  mutate(
    period = recode(period, development = "Development", confirmation = "Jan-Jun later outcomes"),
    label_text = paste0(process, "\n", favourable_months, "/6 months favourable"),
    label_text = factor(
      label_text,
      levels = rev(paste0(levels(f5_transfer$process), "\n",
                          f5_transfer$favourable_months[match(levels(f5_transfer$process), f5_transfer$process)],
                          "/6 months favourable"))
    )
  )

p_f5_transfer <- ggplot(
  f5_transfer_long,
  aes(x = improvement, y = label_text, colour = period, shape = period)
) +
  geom_vline(xintercept = 0, linetype = "dotted", linewidth = 0.45, colour = "#555555") +
  geom_segment(
    data = f5_transfer,
    aes(
      x = development, xend = confirmation,
      y = paste0(process, "\n", favourable_months, "/6 months favourable"),
      yend = paste0(process, "\n", favourable_months, "/6 months favourable")
    ),
    inherit.aes = FALSE, colour = "#BAC2CC", linewidth = 0.8
  ) +
  geom_point(size = 2.25) +
  facet_wrap(~metric_family, scales = "free", ncol = 2, labeller = label_wrap_gen(width = 30)) +
  scale_colour_manual(values = c(
    Development = "#777777",
    `Jan-Jun later outcomes` = pal[["both"]]
  )) +
  labs(
    title = "A  Do historical profiles predict later process outcomes?",
    subtitle = "Positive values favour the profile; all four improve in every later month",
    x = "Improvement in later prediction", y = NULL, colour = NULL, shape = NULL
  ) +
  theme_thesis(9.0)

f5_diagnostics <- read_csv(
  file.path(src_dir, "F5_stability_tier_diagnostics.csv"),
  show_col_types = FALSE
)
f5_tiers <- f5_diagnostics %>%
  filter(group == "Tier outcome") %>%
  mutate(level = factor(level, levels = c("Low", "Medium", "High")))

p_f5_tiers <- ggplot(f5_tiers, aes(x = level, y = value, group = metric, colour = level)) +
  geom_line(colour = "#AEB8C5", linewidth = 0.75) +
  geom_point(size = 2.4) +
  geom_text(
    aes(label = if_else(display_unit == "percent", sprintf("%.2f%%", value), sprintf("%.2f d", value))),
    nudge_y = 0.42, size = 2.75, colour = pal[["dark"]], show.legend = FALSE
  ) +
  facet_wrap(~metric, scales = "free_y", ncol = 2) +
  scale_colour_manual(values = c(
    Low = pal[["favourable"]], Medium = pal[["amber"]], High = pal[["adverse"]]
  )) +
  scale_y_continuous(expand = expansion(mult = c(0.03, 0.24))) +
  labs(
    title = "B  Day-to-day stability and simpler risk tiers",
    subtitle = "Across 16 later-process finalists: median daily rank correlation 0.9967; tier changes 0.37%;\nsame-tier persistence 98.33%",
    x = NULL, y = NULL, colour = NULL
  ) +
  theme_thesis(8.7) +
  theme(legend.position = "none", strip.background = element_blank())

p_f5_standalone <- p_f5_transfer / p_f5_tiers + plot_layout(heights = c(1.05, 1.05))
save_pdf(p_f5_standalone, "f05_standalone_profile_transfer.pdf", height = 5.35)
save_pdf(
  p_f5_transfer + labs(title = "Do retained profiles predict later process outcomes?"),
  "f05_standalone_profile_validity.pdf", height = 3.35
)
save_pdf(
  p_f5_tiers + labs(title = "Day-to-day stability and communication tiers"),
  "f05_profile_stability_tiers.pdf", height = 3.15
)

# -----------------------------------------------------------------------------
# Figure 6.1: direct promise augmentation across breach model families
# -----------------------------------------------------------------------------
robust_root <- file.path(root, "analysis/direct_model_family_robustness_v1")

breach <- read_csv(
  file.path(robust_root, "FIGURE_DATA_BREACH_MODEL_FAMILIES.csv"),
  show_col_types = FALSE
) %>%
  filter(family_status == "evaluated") %>%
  mutate(
    family_label = recode(
      family,
      logistic_l2 = "L2 Logistic Regression",
      random_forest = "Random Forest",
      xgboost = "XGBoost"
    ),
    family_label = factor(
      family_label,
      levels = rev(c("L2 Logistic Regression", "Random Forest", "XGBoost"))
    ),
    block_label = factor(unname(block_labels[profile_block]), levels = c("Seller", "State-OD", "Both")),
    delta_x1000 = 1000 * median_delta_log_loss,
    calibration = if_else(calibration_not_systematically_worse, "Calibration acceptable", "Calibration deteriorated"),
    result_label = paste0(both_improved_month_count, "/6 months")
  )
write_source(breach, "F6_direct_breach_model_families.csv")

p_f6 <- ggplot(breach, aes(x = delta_x1000, y = family_label, colour = block_label)) +
  geom_vline(xintercept = 0, linewidth = 0.5, colour = "#555555") +
  geom_segment(aes(x = 0, xend = delta_x1000, yend = family_label), linewidth = 0.8, alpha = 0.65) +
  geom_point(size = 3.0) +
  geom_text(aes(x = 5.0, label = result_label), hjust = 0, size = 2.9, colour = "#222222") +
  facet_wrap(~block_label, nrow = 1) +
  scale_colour_manual(values = block_colours, guide = "none") +
  coord_cartesian(xlim = c(-22, 15), clip = "off") +
  scale_x_continuous(breaks = c(-20, -10, 0, 10)) +
  labs(
    title = "Historical profiles change missed-date forecasts differently by model",
    subtitle = "Median monthly profile-minus-promise-only log loss; negative improves",
    x = "Median Delta log loss (x 1000)", y = NULL, fill = NULL
  ) +
  theme_thesis(9.0) +
  theme(
    panel.spacing = grid::unit(5, "mm"),
    strip.text = element_text(face = "bold", size = 10.5),
    axis.text.y = element_text(size = 9.0),
    legend.position = "bottom",
    legend.text = element_text(size = 8.0)
  )
save_pdf(p_f6, "f06_direct_breach_model_families.pdf", height = 3.45)

# -----------------------------------------------------------------------------
# Figure 6.2: direct-promise probability reliability
# -----------------------------------------------------------------------------
direct_reliability <- read_csv(
  file.path(root, "analysis/direct_promise_profile_extension_v1/DIRECT_BREACH_RELIABILITY_BINS.csv"),
  show_col_types = FALSE
) %>%
  filter(
    period == "aggregate", cohort == "later_pooled",
    probability_type == "calibrated",
    model_family %in% c("logistic_l2", "xgboost"),
    model_id %in% c("DP0", "DPS", "DPG", "DPB")
  ) %>%
  mutate(
    family_label = recode(
      model_family,
      logistic_l2 = "L2 Logistic Regression",
      xgboost = "XGBoost"
    ),
    model_label = recode(
      model_id,
      DP0 = "Promise only",
      DPS = "+ Seller",
      DPG = "+ State-OD",
      DPB = "+ Both"
    ),
    model_label = factor(
      model_label,
      levels = c("Promise only", "+ Seller", "+ State-OD", "+ Both")
    )
  )
write_source(direct_reliability, "F7_direct_promise_reliability.csv")

reliability_colours <- c(
  `Promise only` = "#6B7280", `+ Seller` = pal[["seller"]],
  `+ State-OD` = pal[["state_od"]], `+ Both` = pal[["both"]]
)

p_f7 <- ggplot(
  direct_reliability,
  aes(x = mean_probability, y = prevalence, colour = model_label, group = model_label)
) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.55, colour = "#555555") +
  geom_point(aes(size = n), alpha = 0.9) +
  facet_wrap(~family_label, nrow = 1) +
  scale_colour_manual(values = reliability_colours) +
  scale_size_continuous(range = c(1.3, 2.4), guide = "none") +
  coord_equal(xlim = c(0, 0.30), ylim = c(0, 0.30)) +
  scale_x_continuous(labels = percent_format(accuracy = 5)) +
  scale_y_continuous(labels = percent_format(accuracy = 5)) +
  labs(
    title = "Do stated missed-date probabilities match observed frequencies?",
    subtitle = "Above the diagonal: risk under-predicted; below: risk over-predicted",
    x = "Mean predicted missed-date probability",
    y = "Observed missed-date frequency",
    colour = NULL
  ) +
  theme_thesis(8.6) +
  theme(legend.position = "bottom", legend.text = element_text(size = 7.7))

save_pdf(p_f7, "f07_direct_promise_reliability.pdf", height = 3.85)

# -----------------------------------------------------------------------------
# Figure 6.3: conditional severity skill and pooled Q90 coverage
# -----------------------------------------------------------------------------
severity <- read_csv(
  file.path(robust_root, "FIGURE_DATA_SEVERITY_MODEL_FAMILIES.csv"),
  show_col_types = FALSE
) %>%
  mutate(
    family_label = recode(
      family,
      linear_quantile = "Linear Quantile",
      random_forest_leaf_weighted_quantile = "Quantile Random Forest",
      xgboost_quantile = "XGBoost Quantile",
      lognormal_ridge = "Lognormal Ridge"
    ),
    family_label = factor(
      family_label,
      levels = rev(c("Linear Quantile", "Quantile Random Forest", "XGBoost Quantile", "Lognormal Ridge"))
    ),
    block_label = factor(unname(block_labels[profile_block]), levels = c("Seller", "State-OD", "Both")),
    quantile_label = if_else(quantile == 0.5, "A  Q50 median skill", "B  Q90 median skill"),
    skill_pct = 100 * median_skill,
    pattern = if_else(evidence_label == "Supported", "Consistent improvement", "Other result")
  )
write_source(severity, "F8A_severity_model_family_skill.csv")

skill_plot <- function(q_value, title_text) {
  ggplot(filter(severity, quantile == q_value),
         aes(x = skill_pct, y = family_label, colour = block_label)) +
    geom_vline(xintercept = 0, linewidth = 0.45, colour = "#555555") +
    geom_segment(
      aes(x = 0, xend = skill_pct, yend = family_label),
      position = position_dodge(width = 0.55), linewidth = 0.55, alpha = 0.55
    ) +
    geom_point(
      aes(fill = pattern), shape = 21, size = 2.6, stroke = 0.8,
      position = position_dodge(width = 0.55)
    ) +
    scale_colour_manual(values = block_colours) +
    scale_fill_manual(values = c(`Consistent improvement` = pal[["dark"]], `Other result` = "white")) +
    labs(title = title_text, x = "Median monthly pinball skill (%)", y = NULL, colour = "Profile block", fill = "Pattern") +
    theme_thesis(8.5) +
    theme(legend.position = "none")
}

p_f8a <- skill_plot(0.5, "A  Conditional median (Q50)")
p_f8b <- skill_plot(0.9, "B  Upper tail (Q90)")

coverage <- read_csv(
  file.path(robust_root, "SEVERITY_MODEL_FAMILY_POOLED.csv"),
  show_col_types = FALSE
) %>%
  filter(period == "aggregate", cohort == "later_pooled", quantile == 0.9) %>%
  distinct(family, model_id, .keep_all = TRUE) %>%
  mutate(
    family_label = recode(
      family,
      linear_quantile = "Linear Quantile",
      random_forest_leaf_weighted_quantile = "Quantile Random Forest",
      xgboost_quantile = "XGBoost Quantile",
      lognormal_ridge = "Lognormal Ridge"
    ),
    family_label = factor(
      family_label,
      levels = rev(c("Linear Quantile", "Quantile Random Forest", "XGBoost Quantile", "Lognormal Ridge"))
    ),
    block_label = recode(model_id, DQ0 = "Promise only", DQS = "Seller", DQG = "State-OD", DQB = "Both"),
    block_label = factor(block_label, levels = c("Promise only", "Seller", "State-OD", "Both"))
  )
write_source(coverage, "F8B_severity_pooled_q90_coverage.csv")

coverage_colours <- c(
  `Promise only` = "#6B7280", Seller = pal[["seller"]],
  `State-OD` = pal[["state_od"]], Both = pal[["both"]]
)
p_f8c <- ggplot(coverage, aes(x = empirical_coverage, y = family_label, colour = block_label)) +
  geom_vline(xintercept = 0.9, linewidth = 0.55, linetype = "dashed", colour = pal[["adverse"]]) +
  geom_point(size = 2.2, position = position_dodge(width = 0.45)) +
  scale_colour_manual(values = coverage_colours) +
  scale_x_continuous(labels = percent_format(accuracy = 1), limits = c(0.825, 0.902), breaks = c(0.83, 0.85, 0.87, 0.89)) +
  labs(
    title = "C  Later-pooled Q90 coverage",
    subtitle = "All 16 specifications remain below nominal 90%",
    x = "Empirical coverage", y = NULL, colour = NULL
  ) +
  theme_thesis(8.5) +
  theme(legend.position = "bottom", legend.text = element_text(size = 7.4))

p_f8 <- (p_f8a | p_f8b) / p_f8c + plot_layout(heights = c(1.0, 0.92))
save_pdf(p_f8, "f08_conditional_severity_model_families.pdf", height = 5.25)
save_pdf((p_f8a + theme(legend.position = "bottom") |
          p_f8b + theme(legend.position = "bottom")) +
           plot_layout(guides = "collect") & theme(legend.position = "bottom"),
         "f08a_conditional_severity_skill.pdf", height = 3.5)
save_pdf(p_f8c + labs(title = "Do Q90 forecasts cover 90% of later positive delays?"),
         "f08b_conditional_q90_coverage.pdf", height = 2.85)
if (final_reader_patch) {
  message("Updated three Chapter 6 presentation PDFs only; no empirical source writes or model fits.")
  quit(save = "no", status = 0)
}

# -----------------------------------------------------------------------------
# Figure 6.3: July-August terminal-regime stress
# -----------------------------------------------------------------------------
terminal <- read_csv(
  file.path(robust_root, "TERMINAL_MODEL_FAMILY_ROBUSTNESS.csv"),
  show_col_types = FALSE
)

regime <- tibble::tribble(
  ~metric, ~period, ~value, ~later_reference, ~display,
  "Breach prevalence", "Jan-Jun", 0.086342190216, 0.086342190216, "8.63%",
  "Breach prevalence", "Jul-Aug", 0.0480530902694, 0.086342190216, "4.81%",
  "Mean positive lateness", "Jan-Jun", 10.6450963474, 10.6450963474, "10.65 d",
  "Mean positive lateness", "Jul-Aug", 5.80532445923, 10.6450963474, "5.81 d"
) %>%
  mutate(relative_to_later = value / later_reference)
write_source(regime, "F9A_terminal_regime_context.csv")

regime_display <- regime %>% mutate(
  actual_value = if_else(metric == "Breach prevalence", 100 * value, value),
  metric_units = if_else(metric == "Breach prevalence", "Breach prevalence (%)",
                        "Mean positive lateness (days)"))
p_f9a <- ggplot(regime_display, aes(x = period, y = actual_value, fill = period)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = display), vjust = -0.35, size = 2.7, colour = pal[["dark"]]) +
  facet_wrap(~metric_units, ncol = 1, scales = "free_y") +
  scale_fill_manual(values = c(`Jan-Jun` = pal[["dark"]], `Jul-Aug` = pal[["amber"]]), guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.20)), breaks = pretty_breaks(3)) +
  labs(
    title = "A  Regime context",
    subtitle = "Observed rates and delays",
    x = NULL, y = NULL, fill = NULL
  ) +
  theme_thesis(8.2) +
  theme(legend.position = "none", axis.text.x = element_text(size = 7.4))

terminal_breach <- terminal %>%
  filter(task == "breach", metric == "delta_log_loss", model_id != "DP0") %>%
  mutate(
    family_label = recode(family, logistic_l2 = "Logistic", random_forest = "RF", xgboost = "XGBoost"),
    family_label = factor(family_label, levels = rev(c("Logistic", "RF", "XGBoost"))),
    profile_block = recode(model_id, DPS = "Seller", DPG = "State-OD", DPB = "Both"),
    profile_block = factor(profile_block, levels = c("Seller", "State-OD", "Both")),
    delta_x1000 = 1000 * estimate
  )
write_source(terminal_breach, "F9B_terminal_breach_model_families.csv")

p_f9b <- ggplot(terminal_breach, aes(x = delta_x1000, y = family_label, colour = profile_block)) +
  geom_vline(xintercept = 0, linewidth = 0.45, colour = "#555555") +
  geom_segment(
    aes(x = 0, xend = delta_x1000, yend = family_label),
    position = position_dodge(width = 0.52), linewidth = 0.55, alpha = 0.6
  ) +
  geom_point(size = 2.2, position = position_dodge(width = 0.52)) +
  scale_colour_manual(values = block_colours) +
  labs(
    title = "B  Breach transfer",
    subtitle = "Negative values improve on promise-only",
    x = "Delta log loss (x 1000)", y = NULL, colour = NULL
  ) +
  theme_thesis(8.2) +
  theme(legend.position = "bottom", legend.text = element_text(size = 7.2))

terminal_q90 <- terminal %>%
  filter(task == "severity", metric == "pinball_skill", quantile == 0.9, model_id != "DQ0") %>%
  mutate(
    family_label = recode(
      family,
      linear_quantile = "Linear",
      random_forest_leaf_weighted_quantile = "QRF",
      xgboost_quantile = "XGBQ",
      lognormal_ridge = "Lognormal"
    ),
    family_label = factor(family_label, levels = rev(c("Linear", "QRF", "XGBQ", "Lognormal"))),
    profile_block = recode(model_id, DQS = "Seller", DQG = "State-OD", DQB = "Both"),
    profile_block = factor(profile_block, levels = c("Seller", "State-OD", "Both")),
    skill_pct = 100 * estimate
  )
write_source(terminal_q90, "F9C_terminal_q90_model_families.csv")

p_f9c <- ggplot(terminal_q90, aes(x = skill_pct, y = family_label, colour = profile_block)) +
  geom_vline(xintercept = 0, linewidth = 0.45, colour = "#555555") +
  geom_segment(
    aes(x = 0, xend = skill_pct, yend = family_label),
    position = position_dodge(width = 0.52), linewidth = 0.55, alpha = 0.6
  ) +
  geom_point(size = 2.2, position = position_dodge(width = 0.52)) +
  scale_colour_manual(values = block_colours) +
  labs(
    title = "C  Q90 transfer",
    subtitle = "Positive skill improves on promise-only",
    x = "Pinball skill (%)", y = NULL, colour = NULL
  ) +
  theme_thesis(8.2) +
  theme(legend.position = "bottom", legend.text = element_text(size = 7.2))

p_f9 <- p_f9a | p_f9b | p_f9c
p_f9 <- p_f9 + plot_layout(widths = c(0.78, 1.05, 1.18), guides = "collect") &
  theme(legend.position = "bottom")
save_pdf(p_f9, "f09_terminal_model_family_transfer.pdf", height = 4.15)

if (v4_probability_only) {
  message("Updated Figure 15 typography only; source CSVs and other figures unchanged.")
} else {
  message("Created selected Chapter 5/6 integration figures; source CSV writing follows mode flags.")
}
