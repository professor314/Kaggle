# Minimal R notebook — summary stats helper.
# Earns the "R Coder" badge and shows a basic R data-summary workflow.

summarise_numeric <- function(df) {
  num <- df[sapply(df, is.numeric)]
  data.frame(
    column = names(num),
    mean   = sapply(num, mean, na.rm = TRUE),
    sd     = sapply(num, sd,   na.rm = TRUE),
    na     = sapply(num, function(x) sum(is.na(x)))
  )
}

# Demo on the built-in mtcars dataset.
print(summarise_numeric(mtcars))
