#!/usr/bin/env Rscript

library("tidyverse")

files <- list.files(path = "out/", pattern = "matmul*")

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
        time = as.numeric(time) * 1e3,
        implementation = as.vector(str_match(bin, "[^/]*(?=/[^/]*$)")),
        compiler = as.vector(str_match(bin, "[^/]*(?=/[^/]*/[^/]*$)")),
        version = as.vector(str_match(bin, "[^/]*$")))

for (m in unique(data$machine)) {
    on_machine <- data %>% filter(machine == m)
    for (c in unique(on_machine$compiler)) {
        on_machine_compiler <- on_machine %>% filter(compiler == c)

        for (s in unique(on_machine_compiler$size)) {
            local <- on_machine_compiler %>%
                filter(size == s) %>%
                group_by(bin, implementation, size) %>%
                reframe(version = unique(version), time) %>%
                mutate(time = time / mean(size ^ 3))

            plot <-
                ggplot(
                    local,
                    aes(x = implementation, y = time)) +
                geom_boxplot(
                    position = "dodge2",
                    outlier.shape = NA,
                    aes(fill = version, color = version)) +
                scale_color_hue(l = 50) +
                ylab("runtime per N^3 [ps]") +
                theme(legend.position = "bottom")

            lims <- boxplot.stats(local$time)$stats[c(1, 5)]
            lims <- c(0, 1 * (lims[2] - lims[1]) + lims[1])

            plot <- plot + coord_cartesian(ylim = lims)

            if (!dir.exists("plots/matmul/"))
                dir.create("plots/matmul/", recursive = TRUE)

            ggsave(
                paste0("plots/matmul/", m, "-", c, "-", s, ".pdf"),
                plot,
                width = 4,
                height = 3)
        }
    }
}
