/* acestep_audio_tool: the command line behind tests/test_acestep_audio.py. It does nothing but call acestep_audio.h.
 *
 *   acestep_audio_tool load     FILE OUT [MAX_SECONDS]   decode FILE (by path); OUT gets the interleaved float32 samples
 *   acestep_audio_tool mem      FILE OUT [MAX_SECONDS]   the same, from the bytes in memory
 *   acestep_audio_tool stereo48k FILE OUT [MAX_SECONDS]  decode, then upstream's _normalize_audio_to_stereo_48k;
 *                                                        OUT gets the planar [2][n] float32 samples
 *
 * One line on stdout: "OK rate=R channels=C frames=N [out_frames=M]" (exit 0) or "ERR message" (exit 1).
 */
#include "acestep_audio.h"

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s load|mem|stereo48k FILE OUT [MAX_SECONDS]\n", argv[0]); return 2; }
    const char *mode = argv[1], *path = argv[2], *outp = argv[3];
    const double maxs = argc > 4 ? atof(argv[4]) : ACA_DEFAULT_MAX_SECONDS;
    char err[512] = "";
    float *pcm = NULL;
    size_t frames = 0;
    int rate = 0, ch = 0, rc;
    if (!strcmp(mode, "mem")) {
        FILE *f = fopen(path, "rb");
        if (!f) { printf("ERR cannot open %s\n", path); return 1; }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        unsigned char *buf = (unsigned char *)malloc((size_t)sz + 1);
        size_t got = fread(buf, 1, (size_t)sz, f);
        fclose(f);
        rc = aca_load_mem_ex(buf, got, maxs, &pcm, &frames, &rate, &ch, err, sizeof err);
        free(buf);
    } else {
        rc = aca_load_ex(path, maxs, &pcm, &frames, &rate, &ch, err, sizeof err);
    }
    if (rc) { printf("ERR %s\n", err); return 1; }
    const float *data = pcm;
    size_t count = frames * (size_t)ch;
    float *norm = NULL;
    size_t nout = 0;
    if (!strcmp(mode, "stereo48k")) {
        if (aca_stereo48k_err(pcm, frames, rate, ch, &norm, &nout, err, sizeof err)) { printf("ERR %s\n", err); free(pcm); return 1; }
        data = norm;
        count = 2 * nout;
    }
    FILE *o = fopen(outp, "wb");
    if (!o) { printf("ERR cannot write %s\n", outp); return 1; }
    fwrite(data, sizeof(float), count, o);
    fclose(o);
    if (norm) printf("OK rate=%d channels=%d frames=%zu out_frames=%zu\n", rate, ch, frames, nout);
    else printf("OK rate=%d channels=%d frames=%zu\n", rate, ch, frames);
    free(pcm);
    free(norm);
    return 0;
}
