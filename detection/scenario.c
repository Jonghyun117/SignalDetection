/* detection/scenario.c — Minimal flat-JSON scenario parser (no dependencies) */
#include "scenario.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* Skip whitespace. */
static const char *skip_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

/* Find the string value for a given key in a flat JSON object.
 * Returns pointer to first char of value, or NULL if not found.
 * value_end is set to one-past-end of the raw value token. */
static const char *find_value(const char *json, const char *key,
                               const char **value_end)
{
    char needle[128];
    snprintf(needle, sizeof needle, "\"%s\"", key);

    const char *pos = strstr(json, needle);
    if (!pos) return NULL;

    pos += strlen(needle);
    pos  = skip_ws(pos);
    if (*pos != ':') return NULL;
    pos++;
    pos = skip_ws(pos);
    if (!*pos) return NULL;

    /* Identify token end. */
    const char *start = pos;
    if (*pos == '"') {
        pos++;                          /* skip opening quote */
        while (*pos && *pos != '"') pos++;
        if (*pos == '"') pos++;
        *value_end = pos;
        return start + 1;              /* skip opening quote in returned ptr */
    } else {
        /* number, true, false, null */
        while (*pos && *pos != ',' && *pos != '}' && *pos != '\n') pos++;
        *value_end = pos;
        return start;
    }
}

int scenario_load(const char *path, Scenario *out)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    rewind(f);

    char *buf = (char *)malloc((size_t)(len + 1));
    if (!buf) { fclose(f); return -1; }

    fread(buf, 1, (size_t)len, f);
    buf[len] = '\0';
    fclose(f);

    /* Defaults. */
    out->rf_center_hz = 0.0;
    out->label[0]     = '\0';

    /* rf_center_hz */
    const char *vend;
    const char *v = find_value(buf, "rf_center_hz", &vend);
    if (!v) { free(buf); return -2; }
    out->rf_center_hz = strtod(v, NULL);

    /* label (optional) */
    v = find_value(buf, "label", &vend);
    if (v && vend > v) {
        size_t n = (size_t)(vend - v - 1);   /* exclude closing quote */
        if (n >= SCENARIO_LABEL_MAX) n = SCENARIO_LABEL_MAX - 1;
        memcpy(out->label, v, n);
        out->label[n] = '\0';
    }

    free(buf);
    return 0;
}

void scenario_print(const Scenario *s)
{
    printf("Scenario: %s\n", s->label[0] ? s->label : "(unlabeled)");
    printf("  RF center: %.3f MHz\n", s->rf_center_hz / 1e6);
}
