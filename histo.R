#!/usr/bin/env Rscript

library("tidyverse")

files <- list.files(path = "out/", pattern = "histo*")

data <- lapply(
    files,
    function(x) {
        read.csv(
            paste0("out/", x),
            header = FALSE,
            col.names = c("bin", "input", "size", "machine", "date", "time"))
    })
data <- do.call("rbind", data)

data <- data %>%
    mutate(
        time = as.numeric(time) / 1e9,
        kind = as.vector(str_match(bin, "[^/]*(?=/[^/]*$)")),
        compiler = as.vector(str_match(bin, "[^/]*(?=/[^/]*/[^/]*$)")),
        subkind = as.vector(str_match(bin, "[^/]*$")))

for (m in unique(data$machine)) {
    on_machine <- data %>% filter(as.vector(machine == m))
    for (c in unique(on_machine$compiler)) {
        local <- on_machine %>% filter(as.vector(compiler == c))

        local <- local %>%
            group_by(bin, kind, size) %>%
            reframe(time = mean(time), subkind = unique(subkind)) %>%
            group_by(kind, size) %>%
            mutate(relative.time = time / time[subkind == "plain"]) %>%
            mutate(x = paste0(2, "^", log2(size)))

        plot <- ggplot(local, aes(x = x, y = relative.time, fill = subkind)) +
            geom_bar(stat = "identity", position = "dodge") +
            facet_grid(. ~ kind) +
            xlab("input text size") +
            ylab("relative runtime") +
            theme(
                axis.text.x = element_text(angle = 45, hjust = 1),
                legend.position = "bottom")

        if (!dir.exists("plots/histo/"))
            dir.create("plots/histo/", recursive = TRUE)

        ggsave(paste0("plots/histo/", m, "-", c, ".pdf"), plot)
    }
}
