/* detection/scenario.h — Scenario file (JSON) parser */
#ifndef SCENARIO_H
#define SCENARIO_H

#define SCENARIO_LABEL_MAX 64

typedef struct {
    double rf_center_hz;            /* RF tuning center frequency (Hz)     */
    char   label[SCENARIO_LABEL_MAX]; /* human-readable label              */
} Scenario;

/* Parse a JSON scenario file.
 * Returns 0 on success, -1 on file error, -2 on parse error. */
int scenario_load(const char *path, Scenario *out);

/* Print scenario to stdout. */
void scenario_print(const Scenario *s);

#endif /* SCENARIO_H */
