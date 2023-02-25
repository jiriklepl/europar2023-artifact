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
        algorithm = as.vector(str_match(bin, "[^/]*(?=/[^/]*$)")),
        compiler = as.vector(str_match(bin, "[^/]*(?=/[^/]*/[^/]*$)")),
        implementation = as.vector(str_match(bin, "[^/]*$")))

for (m in unique(data$machine)) {
    on_machine <- data %>% filter(machine == m)
    for (c in unique(on_machine$compiler)) {
        on_machine_compiler <- on_machine %>% filter(compiler == c)

        for (s in unique(on_machine_compiler$size)) {
            local <- on_machine_compiler %>%
                filter(size == s) %>%
                group_by(bin, algorithm, size) %>%
                reframe(implementation = unique(implementation), time) %>%
                mutate(time = time / mean(size ^ 3))

            plot <- ggplot(local, aes(x = algorithm, y = time, fill = implementation)) +
                geom_boxplot(position = "dodge", outlier.shape = NA) +
                ylab("runtime per N^3 [ps]") +
                ylim(0, max(local$time)) +
                theme(legend.position = "bottom")

            if (!dir.exists("plots/matmul/"))
                dir.create("plots/matmul/", recursive = TRUE)

            ggsave(paste0("plots/matmul/", m, "-", c, "-", s, ".pdf"), plot)
        }
    }
}
