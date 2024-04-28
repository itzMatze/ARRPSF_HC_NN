/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Debug/PixelDebug.h"
#include "Rendering/Lights/LightBVHSampler.h"
#include "Rendering/Lights/EnvMapSampler.h"

using namespace Falcor;

/**
 * Compute path tracer.
 */
class ComputePathTracer : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ComputePathTracer, "ComputePathTracer", "Compute path tracer.");

    static ref<ComputePathTracer> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<ComputePathTracer>(pDevice, props);
    }

    ComputePathTracer(ref<Device> pDevice, const Properties& props);
    void reset();

    virtual void setProperties(const Properties& props) override;
    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return mpPixelDebug->onMouseEvent(mouseEvent); }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;

private:
    void parseProperties(const Properties& props);
    void createPasses(const RenderData& renderData);
    void setupData(RenderContext* pRenderContext);
    void setupBuffers();
    void bindData(const RenderData& renderData, uint2 frameDim);

    enum // Buffer
    {
        HASH_ENTRIES_BUFFER = 0,
        HASH_CACHE_VOXEL_DATA_BUFFER_0 = 1,
        HASH_CACHE_VOXEL_DATA_BUFFER_1 = 2,
        NN_PRIMAL_BUFFER = 3,
        NN_FILTERED_PRIMAL_BUFFER = 4,
        NN_GRADIENT_BUFFER = 5,
        NN_GRADIENT_AUX_BUFFER = 6,
        LOSS_SUM_BUFFER = 7,
        BUFFER_COUNT
    };

    enum // Passes
    {
        TRAIN_NN_FILL_CACHE_PASS = 0,
        RESOLVE_PASS = 1,
        PATH_TRACING_PASS = 2,
        NN_GRADIENT_CLEAR_PASS = 3,
        NN_GRADIENT_DESCENT_PASS = 4,
        PASS_COUNT
    };

    // Current scene.
    float3 mCamPos;
    ref<Scene> mpScene;
    ref<SampleGenerator> mpSampleGenerator;

    // show contribution of specific bounce range, upper bound will terminate path
    uint mLowerBounceCount = 0;
    uint mUpperBounceCount = 20;
    bool mUseNEE = true;
    bool mUseMIS = true;
    bool mMISUsePowerHeuristic = true;
    bool mUseRR = true;
    // Use importance sampling for materials.
    bool mUseImportanceSampling = true;
    // starting value for the survival probability of russian roulette
    float mRRProbStartValue = 1.2f;
    // factor by which the survival probability gets reduced
    float mRRProbReductionFactor = 0.9f;
    bool mDebugPathLength = false;
    mutable LightBVHSampler::Options mLightBVHOptions;

    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;
    std::unique_ptr<EmissiveLightSampler> mpEmissiveSampler;
    ref<ParameterBlock> mpSamplerBlock;
    std::unique_ptr<PixelDebug> mpPixelDebug;

// Hash Cache
    // should hash cache be enabled, this state is used when the options are applied
    bool mEnableHashCache = true;
    // is hash cache currently used in the program
    bool mHashCacheActive = true;
    uint32_t mHashCacheHashMapSizeExp = 22;
    uint32_t mHashCacheHashMapSize = std::pow(2u, mHashCacheHashMapSizeExp);
    bool mHashCacheDebugVoxels = false;
    bool mHashCacheDebugColor = false;
    bool mHashCacheDebugLevels = false;

// NN
    struct NNParams
    {
        bool enable = true;
        bool active = true;
        enum OptimizerType {
            SGD = 0,
            ADAM = 1,
        };
        struct OptimizerParam {
            OptimizerType type = ADAM;
            int step_count = 0;
            float learn_r = 0.01;
            float param_0 = 0.9;
            float param_1 = 0.999;
            float param_2 = 1e-08;
        } optimizerParams;
        int nnLayerCount = 5;
        Gui::DropdownList nnLayerWidthList{Gui::DropdownValue{16, "16"}, Gui::DropdownValue{32, "32"}};
        uint nnLayerWidth = 32;
        uint nnParamCount = 0;
        int gradOffset = 0;
        float2 weightInitBound = float2(0.001, 0.02);
        bool debugOutput = true;

        void update()
        {
            nnParamCount = ((nnLayerWidth * nnLayerWidth /*weights*/ + nnLayerWidth /*biases*/) * nnLayerCount + 120 * 68 * 14 /*feature grid*/);
        }
    } mNNParams;

    uint mFrameCount = 0;
    bool mOptionsChanged = true;

    std::array<ref<Buffer>, BUFFER_COUNT> mBuffers;
    std::array<ref<ComputePass>, PASS_COUNT> mPasses;
};

