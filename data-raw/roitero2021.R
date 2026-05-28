dat <- read.csv("data-raw/dataframe.csv", header = TRUE)
pbapply::pboptions(type = "txt")
roitero2021 <- subset(dat, relevance_scale == "S100")[, setdiff(
  colnames(dat),
  c("normalized_relevance_score", "relevance_scale")
)]
usethis::use_data(roitero2021, overwrite = TRUE)
