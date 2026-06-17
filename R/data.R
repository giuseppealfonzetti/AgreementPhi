#' Download Roitero et al. (2021) crowdsourced relevance assessments
#'
#' Downloads and returns the S100-scale subset of the crowdsourcing experiment
#' in Roitero et al. (2021). The upstream CSV (~48 MB) is fetched from GitHub
#' and parsed into a data frame. When called from the package source tree the
#' local \code{data-raw/dataframe.csv} is used instead, avoiding any download.
#'
#' @param SCALE Scale of interest. Default "S100".
#' @return A data frame with 56 472 rows and 11 columns:
#' \describe{
#'   \item{topic_id}{The ID of the topic.}
#'   \item{unit_id}{The ID of the HIT.}
#'   \item{document_id}{The name of the document.}
#'   \item{gold}{Whether the document is a gold one.}
#'   \item{doc_pos}{Position of the document in the set as seen by the worker.}
#'   \item{worker_id}{Encrypted worker ID (to allow anonymity).}
#'   \item{relevance_score}{Score as submitted by the worker (0--100 continuous scale).}
#'   \item{cumulative_time}{Time spent by the worker to assess the document.}
#'   \item{comment}{Comment left by the worker as justification for the score.}
#'   \item{trec}{Score as submitted by TREC (if available).}
#'   \item{sormunen}{Score as submitted by Sormunen (if available).}
#' }
#'
#' @source \url{https://github.com/KevinRoitero/CrowdsourcingRelevanceScales}
#' @references Roitero, K., Maddalena, E., Mizzaro, S., and Scholer, F. (2021).
#'   On the Effect of Relevance Scales in Crowdsourcing Relevance Assessments
#'   for Information Retrieval Evaluation. *Information Processing and
#'   Management*, 58(6).
#'
#' @examples
#' \dontrun{
#' d <- download_roitero2021()
#' str(d)
#' }
#'
#' @export
download_roitero2021 <- function(SCALE = c("S100", "S4")) {
  SCALE <- match.arg(SCALE)
  local_csv <- file.path("data-raw", "dataframe.csv")
  if (!file.exists(local_csv)) {
    local_csv <- tempfile(fileext = ".csv")
    utils::download.file(
      paste0(
        "https://raw.githubusercontent.com/",
        "KevinRoitero/CrowdsourcingRelevanceScales/main/dataframe.csv"
      ),
      local_csv,
      quiet = TRUE
    )
  }
  raw <- utils::read.csv(local_csv, header = TRUE)
  subset(raw, raw$relevance_scale == SCALE)[,
    setdiff(colnames(raw), c("normalized_relevance_score", "relevance_scale"))
  ]
}
