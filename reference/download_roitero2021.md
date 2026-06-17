# Download Roitero et al. (2021) crowdsourced relevance assessments

Downloads and returns the S100-scale subset of the crowdsourcing
experiment in Roitero et al. (2021). The upstream CSV (~48 MB) is
fetched from GitHub and parsed into a data frame. When called from the
package source tree the local `data-raw/dataframe.csv` is used instead,
avoiding any download.

## Usage

``` r
download_roitero2021(SCALE = c("S100", "S4"))
```

## Source

<https://github.com/KevinRoitero/CrowdsourcingRelevanceScales>

## Arguments

- SCALE:

  Scale of interest. Default "S100".

## Value

A data frame with 56 472 rows and 11 columns:

- topic_id:

  The ID of the topic.

- unit_id:

  The ID of the HIT.

- document_id:

  The name of the document.

- gold:

  Whether the document is a gold one.

- doc_pos:

  Position of the document in the set as seen by the worker.

- worker_id:

  Encrypted worker ID (to allow anonymity).

- relevance_score:

  Score as submitted by the worker (0–100 continuous scale).

- cumulative_time:

  Time spent by the worker to assess the document.

- comment:

  Comment left by the worker as justification for the score.

- trec:

  Score as submitted by TREC (if available).

- sormunen:

  Score as submitted by Sormunen (if available).

## References

Roitero, K., Maddalena, E., Mizzaro, S., and Scholer, F. (2021). On the
Effect of Relevance Scales in Crowdsourcing Relevance Assessments for
Information Retrieval Evaluation. *Information Processing and
Management*, 58(6).

## Examples

``` r
if (FALSE) { # \dontrun{
d <- download_roitero2021()
str(d)
} # }
```
