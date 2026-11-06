/* gemma4_tok_cli: drive gemma4_tok.h from a test.
 *
 *   gemma4_tok_cli tokenizer.json            then, per line on stdin:
 *     E <hex of the UTF-8 text>   ->  the ids, space separated
 *     D <id> <id> ...             ->  hex of the bytes those ids spell, concatenated
 *
 * Hex keeps newlines, NULs and invalid UTF-8 out of the pipe's way. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../gemma4_tok.h"

static int unhex(const char *h, unsigned char *out) {
    int n = 0;
    for (; h[0] && h[1] && h[0] != '\n' && h[0] != '\r'; h += 2) {
        unsigned v; if (sscanf(h, "%2x", &v) != 1) break;
        out[n++] = (unsigned char)v;
    }
    return n;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: gemma4_tok_cli tokenizer.json\n"); return 2; }
    G4Tok T;
    if (!g4_tok_load(&T, argv[1])) return 1;
    size_t cap = 1 << 22;
    char *line = (char *)malloc(cap);
    while (fgets(line, (int)cap, stdin)) {
        if (line[0] == 'E' && line[1] == ' ') {
            unsigned char *text = (unsigned char *)malloc(cap / 2 + 1);
            int len = unhex(line + 2, text);
            int *ids = (int *)malloc(((size_t)len + 8) * sizeof(int));
            int n = g4_tok_encode(&T, (const char *)text, len, ids, len + 8);
            for (int i = 0; i < n; i++) printf(i ? " %d" : "%d", ids[i]);
            printf("\n"); fflush(stdout);
            free(text); free(ids);
        } else if (line[0] == 'D' && line[1] == ' ') {
            char *p = line + 2;
            for (;;) {
                char *end; long id = strtol(p, &end, 10);
                if (end == p) break;
                char piece[512]; int k = g4_tok_piece(&T, (int)id, piece, sizeof piece);
                for (int i = 0; i < k; i++) printf("%02x", (unsigned char)piece[i]);
                p = end;
            }
            printf("\n"); fflush(stdout);
        }
    }
    free(line);
    return 0;
}
