skip_on_cran()
skip_if_offline()
roitero2021 <- download_roitero2021()

for (tid in sort(unique(roitero2021$topic_id))) {
  local({
    tid <- tid
    test_that(paste("inflated fit is valid for topic", tid), {
      d <- subset(roitero2021, topic_id == tid)
      gw <- intersect(
        unique(d[d$relevance_score > 50 & d$gold == "H", ]$unit_id),
        unique(d[d$relevance_score < 50 & d$gold == "N", ]$unit_id)
      )
      d <- subset(d, unit_id %in% gw & gold == "null")
      ratings <- d$relevance_score / 100
      items <- as.integer(factor(d$document_id))
      rd <- rating_data(ratings, items, VERBOSE = FALSE)
      expect_equal(rd$data_type, "inflated")
      fit <- agreement(rd, NUISANCE = "items")
      expect_true(is.finite(fit$modified$agreement))
      expect_true(fit$modified$agreement >= 0 && fit$modified$agreement <= 1)
      ci <- confint(fit)
      expect_true(all(is.finite(ci$agreement)))
      expect_true(all(is.finite(ci$parameters)))
    })
  })
}
