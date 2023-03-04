#!/usr/bin/env Rscript

library("ggplot2")
library("dplyr")
library("stringr")

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
    filter(size >= 2 ** 19) %>%
    filter(size < 2 ** 29) %>%
    mutate(
        time = as.numeric(time),
        implementation = as.vector(str_match(bin, "[^/]*(?=/[^/]*$)")),
        compiler = as.vector(str_match(bin, "[^/]*(?=/[^/]*/[^/]*$)")),
        version = as.vector(str_match(bin, "[^/]*$")))

for (m in unique(data$machine)) {
    on_machine <- data %>% filter(as.vector(machine == m))
    for (c in unique(on_machine$compiler)) {
        local <- on_machine %>%
            filter(
                as.vector(compiler == c),
                implementation != 'cpu-range')

        local <- local %>%
            group_by(implementation, size) %>%
            reframe(time = time / median(time), version, bin) %>%
            group_by(bin, implementation, size) %>%
            reframe(
                relative.time = time,
                version = unique(version),
                )

        plot <-
            ggplot(
                local,
                aes(x = factor(size), y = relative.time, fill = version)) +
            geom_boxplot(position = "dodge2", outlier.shape = NA) +
            geom_hline(yintercept = 1) +
            facet_grid(. ~ implementation) +
            xlab("input text size") +
            ylab("relative runtime") +
            scale_x_discrete(labels=c(expression(2^19), expression(2^21), expression(2^23), expression(2^25), expression(2^27), expression(2^29))) +
            theme(legend.position = "bottom")

        plot <- plot + coord_cartesian(ylim = c(1 - .15, 1 + .10))

        if (!dir.exists("plots/histo/"))
            dir.create("plots/histo/", recursive = TRUE)

        ggsave(
            paste0("plots/histo/", m, "-", c, ".pdf"),
            plot,
            width = 4,
            height = 3)
    }
}
