#!/usr/bin/env Rscript

library('tidyverse')

files <- list.files(path='out',pattern='histo*')

data <- lapply(files, function(x) read.csv(paste('out/', x, sep=''),header=FALSE, col.names=c('bin', 'input', 'size', 'machine', 'date', 'time')))
data <- do.call('rbind', data)

data <- data %>% mutate(time=apply(data['time'], 1, function(x) as.numeric(str_remove(x, ' .*'))))
data <- data %>% mutate(kind=apply(data['bin'], 1, function(x) str_match(str_match(x, '[^/]*/[^/]*$'), '[^-]*-[^-]*')))
data <- data %>% mutate(subkind=apply(data['bin'], 1, function(x) str_remove(str_match(x, '[^/]*/[^/]*$'), '^[^-]*-[^-]*-')))

machines <- unique(data$machine)

for (i in machines) {
    local <- data %>% filter(machine==i) %>% group_by(bin, kind, size) %>% summarize(time=mean(time), subkind=unique(subkind))
    local <- local %>% group_by(kind, size) %>% mutate(reltime=time/time[subkind == 'plain'])

    local %>% ggplot(aes(x=log2(size), y=reltime, fill=subkind)) +
            geom_bar(stat='identity', position='dodge') +
            facet_grid(. ~ kind) +
            theme(legend.position='bottom')

    ggsave(paste('histo-', i, '-plots.pdf', sep=''))
}
