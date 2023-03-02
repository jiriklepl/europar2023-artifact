#!/usr/bin/env Rscript

library("ggplot2")
library("dplyr")
library("stringr")

files <- list.files(path = "out/", pattern = "kmeans*")

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
        time = as.numeric(time) / 1e6,
        implementation = as.vector(str_match(bin, "[^/]*(?=/[^/]*$)")),
        compiler = as.vector(str_match(bin, "[^/]*(?=/[^/]*/[^/]*$)")),
        input = as.vector(str_match(input, "[^/]*$")),
        version = as.vector(str_match(bin, "[^/]*$")))

for (m in unique(data$machine)) {
    on_machine <- data %>% filter(machine == m)
    for (c in unique(on_machine$compiler)) {
        if (c == "python")
            next

        on_machine_compiler <- on_machine %>% filter((compiler == c) | (compiler == "python"))
        for (i in unique(on_machine_compiler$input)) {
            local <- on_machine_compiler %>%
                filter(input == i) %>%
                group_by(bin, implementation, input) %>%
                reframe(version = unique(version), time) %>%
                mutate(time = time)

            plot <-
                ggplot(
                    local,
                    aes(x = implementation, y = time)) +
                geom_boxplot(
                    position = "dodge2",
                    outlier.shape = NA,
                    aes(fill = version)) +
                ylab("runtime [ms]") +
                theme(legend.position = "bottom")

            lims <- boxplot.stats(local$time)$stats[c(1, 5)]
            lims <- c(0, 1 * (lims[2] - lims[1]) + lims[1])

            plot <- plot + coord_cartesian(ylim = lims)

            if (!dir.exists("plots/kmeans/"))
                dir.create("plots/kmeans/", recursive = TRUE)

            ggsave(
                paste0("plots/kmeans/", m, "-", c, "-", i, ".pdf"),
                plot,
                width = 4,
                height = 3)
        }
    }
}
