#!/usr/bin/env Rscript

library("tidyverse")

files <- list.files(path = "out", pattern = "histo*")

data <- lapply(
    files,
    function(x) {
        read.csv(
            paste("out/", x, sep = ""),
            header = FALSE,
            col.names = c("bin", "input", "size", "machine", "date", "time"))
    })
data <- do.call("rbind", data)

data <- data %>%
    mutate(
        time = as.numeric(time) / 1e9,
        kind = str_match(str_match(bin, "[^/]*/[^/]*/[^/]*$"), "^[^/]*/[^/]*"),
        subkind = str_match(bin, "[^/]*$"))

machines <- unique(data$machine)

for (i in machines) {
    local <- data %>%
        filter(machine == i) %>%
        group_by(bin, kind, size) %>%
        summarize(time = mean(time), subkind = unique(subkind)) %>%
        group_by(kind, size) %>%
        mutate(relative.time = time / time[subkind == "plain"])

    local %>% ggplot(aes(x = log2(size), y = relative.time, fill = subkind)) +
            geom_bar(stat = "identity", position = "dodge") +
            facet_grid(. ~ kind) +
            theme(legend.position = "bottom")

    ggsave(paste("histo-", i, "-plots.pdf", sep = ""))
}
