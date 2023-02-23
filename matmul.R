#!/usr/bin/env Rscript

library("tidyverse")

files <- list.files(path = "out", pattern = "matmul*")

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
    filter(size == 512)

data <- data %>%
    mutate(bin = str_replace(bin, "(poly)-(.*)2", "\\12-\\2")) %>%
    mutate(
        time = as.numeric(time) / 1e9,
        kind = str_match(str_match(bin, "[^/]*/[^/]*$"), "[^-]*-[^-]*"),
        subkind = str_remove(str_match(bin, "[^/]*/[^/]*$"), "^[^-]*-[^-]*-"))

machines <- unique(data$machine)

for (i in machines) {
    local <- data %>% filter(machine == i) %>% group_by(bin, kind, size)

    local %>%
        reframe(subkind = unique(subkind), time) %>%
        mutate(time = time / mean(size ^ 3)) %>%
        ggplot(aes(x = kind, y = time, fill = subkind)) +
            geom_boxplot(position = "dodge", outlier.shape = NA) +
            ylab("runtime per N^3 [ns]") +
            ylim(0, max(local$time / mean(local$size ^ 3))) +
            theme(legend.position = "bottom")

    ggsave(paste("matmul-", i, "-plots.pdf", sep = ""))
}
