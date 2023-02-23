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
        kind = str_match(str_match(bin, "[^/]*/[^/]*$"), "^[^/]*"),
        compiler = str_match(str_match(bin, "[^/]*/[^/]*/[^/]*$"), "^[^/]*"),
        subkind = str_match(bin, "[^/]*$"))

for (m in unique(data$machine)) {
    on_machine <- data %>% filter(machine == m)
    for (c in unique(on_machine$compiler)) {
        if (length(unique(on_machine$compiler)) > 1)
            local <- on_machine %>% filter(compiler == c)
        else
            local <- on_machine

        local <- local %>%
            group_by(bin, kind, size) %>%
            summarize(time = mean(time), subkind = unique(subkind)) %>%
            group_by(kind, size) %>%
            mutate(relative.time = time / time[subkind == "plain"])

        local %>%
            ggplot(aes(x = log2(size), y = relative.time, fill = subkind)) +
                geom_bar(stat = "identity", position = "dodge") +
                facet_grid(. ~ kind) +
                theme(legend.position = "bottom")

        if (!dir.exists("plots/histo/"))
            dir.create("plots/histo/", recursive = TRUE)

        ggsave(
            paste("plots/histo/", m, "-", c, ".pdf", sep = ""))
    }
}
