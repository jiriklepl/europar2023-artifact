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
    mutate(
        time = as.numeric(time) / 1e9,
        kind = str_match(str_match(bin, "[^/]*/[^/]*$"), "^[^/]*"),
        compiler = str_match(str_match(bin, "[^/]*/[^/]*/[^/]*$"), "^[^/]*"),
        subkind = str_match(bin, "[^/]*$"))

for (m in unique(data$machine)) {
    on_machine <- data %>% filter(machine == m)
    for (c in unique(on_machine$compiler)) {
        on_machine_compiler <- on_machine %>% filter(compiler == c)

        for (s in unique(on_machine_compiler$size)) {
            local <- on_machine_compiler %>%
                filter(size == s) %>%
                group_by(bin, kind, size)

            local %>%
                reframe(subkind = unique(subkind), time) %>%
                mutate(time = time / mean(size ^ 3)) %>%
                ggplot(aes(x = kind, y = time, fill = subkind)) +
                    geom_boxplot(position = "dodge", outlier.shape = NA) +
                    ylab("runtime per N^3 [ns]") +
                    ylim(0, max(local$time / mean(local$size ^ 3))) +
                    theme(legend.position = "bottom")

            if (!dir.exists("plots/matmul/"))
                dir.create("plots/matmul/", recursive = TRUE)

            ggsave(
                paste("plots/matmul/", m, "-", c, "-", s, ".pdf", sep = ""))
        }
    }
}
