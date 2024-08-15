#ifndef _SRENDERER_ADDON_RESERVOIR_
#define _SRENDERER_ADDON_RESERVOIR_

import Rendering.Materials.IMaterialInstance;


struct Reservoir
{
    
    float wSum;
    uint32_t M;
    float targetPdf;
    BSDFSample sample;

    __init()
    {
        wSum = 0;
        M = 0;
        targetPdf = 0;
    }


    [mutating] void update(BSDFSample x, float initPdf, float targetPDF, float u)
    {
        float w = targetPDF / initPdf;
        wSum = wSum + w;
        M += 1;
        if (u < w / wSum)
        {
            sample = x;
            targetPdf = targetPDF;
        }
    }

    float getShadingWeight() { return (wSum / M) / targetPdf; }

    BSDFSample getSample() { return sample; }
};

#endif
