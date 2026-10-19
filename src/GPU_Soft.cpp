/*
    Copyright 2016-2026 melonDS team

    This file is part of melonDS.

    melonDS is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License as published by the Free
    Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    melonDS is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with melonDS. If not, see http://www.gnu.org/licenses/.
*/

#include <chrono>
#include <string.h>
#include <thread>

#include "NDS.h"
#include "GPU_Soft.h"
#include "GPU_ColorOp.h"

namespace melonDS
{

namespace
{
inline void CPURelax()
{
#if defined(__aarch64__) || defined(__arm__)
    asm volatile("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

// busy-wait for a condition, for at most the given time
// cheaper than sleeping when the other thread is expected to be quick;
// after a short while, it lets other threads have the core if they need it
template <typename F>
bool SpinUntil(F&& done, std::chrono::microseconds limit)
{
    constexpr std::chrono::microseconds yieldAfter(20);
    auto start = std::chrono::steady_clock::now();
    bool yield = false;
    for (u32 i = 1;; i++)
    {
        if (done())
            return true;

        if (yield)
            std::this_thread::yield();
        else
            CPURelax();

        if ((i & 0x3F) == 0 || yield)
        {
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (elapsed > limit)
                return done();
            yield = elapsed > yieldAfter;
        }
    }
}

// how long the render thread waits for the next job before going to sleep
constexpr std::chrono::microseconds RenderThreadSpin(500);
// how long the emulation thread waits for the render thread before going to sleep
constexpr std::chrono::microseconds EmuThreadSpin(100);
}

SoftRenderer::SoftRenderer(melonDS::NDS& nds)
    : Renderer(nds.GPU), State2D_A(0, nds.GPU), State2D_B(1, nds.GPU)
{
    // the top and bottom screens of each buffer are contiguous, so that a frontend
    // can show them stacked without copying
    const size_t len = 256 * 192;
    Framebuffer[0][0] = new u32[len * 2];
    Framebuffer[0][1] = Framebuffer[0][0] + len;
    Framebuffer[1][0] = new u32[len * 2];
    Framebuffer[1][1] = Framebuffer[1][0] + len;
    BackBuffer = 0;

    auto rend2D_A = std::make_unique<SoftRenderer2D>(State2D_A, *this);
    auto rend2D_B = std::make_unique<SoftRenderer2D>(State2D_B, *this);
    Soft2D_A = rend2D_A.get();
    Soft2D_B = rend2D_B.get();
    Rend2D_A = std::move(rend2D_A);
    Rend2D_B = std::move(rend2D_B);
    auto rend3D = std::make_unique<SoftRenderer3D>(GPU.GPU3D, *this);
    Soft3D = rend3D.get();
    Rend3D = std::move(rend3D);

    Sema_Work = Platform::Semaphore_Create();
    Sema_Done = Platform::Semaphore_Create();
}

SoftRenderer::~SoftRenderer()
{
    SetThreaded2D(false);

    Platform::Semaphore_Free(Sema_Work);
    Platform::Semaphore_Free(Sema_Done);

    delete[] Framebuffer[0][0];
    delete[] Framebuffer[1][0];
}

void SoftRenderer::Reset()
{
    WaitIdle();

    const size_t len = 256 * 192 * sizeof(u32);
    memset(Framebuffer[0][0], 0, len);
    memset(Framebuffer[0][1], 0, len);
    memset(Framebuffer[1][0], 0, len);
    memset(Framebuffer[1][1], 0, len);

    Rend2D_A->Reset();
    Rend2D_B->Reset();
    Rend3D->Reset();
}

void SoftRenderer::Stop()
{
    WaitIdle();

    // clear framebuffers to black
    const size_t len = 256 * 192 * sizeof(u32);
    memset(Framebuffer[0][0], 0, len);
    memset(Framebuffer[0][1], 0, len);
    memset(Framebuffer[1][0], 0, len);
    memset(Framebuffer[1][1], 0, len);
}


void SoftRenderer::PreSavestate()
{
    WaitIdle();

    auto rend3d = dynamic_cast<SoftRenderer3D*>(Rend3D.get());
    if (rend3d->IsThreaded())
        rend3d->SetupRenderThread();
}

void SoftRenderer::PostSavestate()
{
    auto rend3d = dynamic_cast<SoftRenderer3D*>(Rend3D.get());
    if (rend3d->IsThreaded())
        rend3d->EnableRenderThread();
}


void SoftRenderer::SetRenderSettings(RendererSettings& settings)
{
    auto rend3d = dynamic_cast<SoftRenderer3D*>(Rend3D.get());
    rend3d->SetThreaded(settings.Threaded);
    rend3d->SetSplit(settings.Split3D);

    SetThreaded2D(settings.Threaded2D);
}


void SoftRenderer::DrawScanline(u32 line)
{
    u32 *dstA, *dstB;
    u32 dstoffset = 256 * line;
    if (GPU.ScreenSwap)
    {
        dstA = &Framebuffer[BackBuffer][0][dstoffset];
        dstB = &Framebuffer[BackBuffer][1][dstoffset];
    }
    else
    {
        dstA = &Framebuffer[BackBuffer][1][dstoffset];
        dstB = &Framebuffer[BackBuffer][0][dstoffset];
    }

    // the position used for drawing operations is based on VCOUNT
    u32 vcount = GPU.VCount;
    u32 dispcnt = GPU.GPU2D_A.DispCnt;

    if (Threaded2D)
    {
        Job2D& job = BeginJob();
        job.Sprites = false;
        job.Line = line;
        job.VCount = vcount;
        job.DstA = dstA;
        job.DstB = dstB;
        job.ScreensEnabled = GPU.ScreensEnabled;
        job.Capture = false;

        if (vcount < 192)
        {
            // the 3D renderer's buffer only changes when the next frame starts (see Start3DRendering)
            u32* line3D = Rend3D->GetLine(vcount);
            job.Line3DCopied = !Soft3D->IsStableLine(line3D);
            if (job.Line3DCopied)
                memcpy(job.Line3D, line3D, sizeof(job.Line3D));
            else
                job.Line3DOffset = line3D - Soft3D->LineBuffer();

            BeginJobState(job);
            UpdateBGVRAM(GPU.GPU2D_A);
            UpdateBGVRAM(GPU.GPU2D_B);

            job.BrightnessA = GPU.MasterBrightnessA;
            job.BrightnessB = GPU.MasterBrightnessB;

            job.HasDisplayLine = false;
            switch ((dispcnt >> 16) & 0x3)
            {
            case 2: // VRAM display
                {
                    u32 vrambank = (dispcnt >> 18) & 0x3;
                    if (GPU.VRAMMap_LCDC & (1<<vrambank))
                    {
                        SyncCaptures(vrambank);
                        memcpy(job.DisplayLine, &((u16*)GPU.VRAM[vrambank])[vcount * 256], sizeof(job.DisplayLine));
                        job.HasDisplayLine = true;
                    }
                }
                break;

            case 3: // FIFO display
                memcpy(job.DisplayLine, GPU.DispFIFOBuffer, sizeof(job.DisplayLine));
                job.HasDisplayLine = true;
                break;
            }

            u32 captureCnt = GPU.CaptureCnt;
            if (GPU.CaptureEnable && CaptureNeeded(vcount, captureCnt, GPU.VRAMMap_LCDC))
            {
                job.Capture = true;
                job.CaptureCnt = captureCnt;
                job.HasCaptureB = false;

                if ((captureCnt >> 29) & 0x3)
                {
                    // source B may be a VRAM bank that is still being captured to
                    u32 srcvram = (dispcnt >> 18) & 0x3;
                    if (!(captureCnt & (1<<25)) && (GPU.VRAMMap_LCDC & (1<<srcvram)))
                        SyncCaptures(srcvram);

                    const u16* srcB = CaptureSourceB(vcount, captureCnt, dispcnt);
                    if (srcB)
                    {
                        u32 width = (((captureCnt >> 20) & 0x3) == 0) ? 128 : 256;
                        memcpy(job.CaptureB, srcB, width * sizeof(u16));
                        job.HasCaptureB = true;
                    }
                }

                CaptureJobs[(captureCnt >> 16) & 0x3] = JobsSubmitted.load(std::memory_order_relaxed) + 1;
                CaptureLCDCMap = GPU.VRAMMap_LCDC;
            }
        }

        SubmitJob();
        if (job.Capture && CaptureImmediate)
            WaitIdle();
        return;
    }

    const u16* displayLine = nullptr;
    if (vcount < 192)
    {
        // retrieve 3D output
        Output3D = Rend3D->GetLine(vcount);

        static_cast<GPU2DState&>(State2D_A) = GPU.GPU2D_A;
        static_cast<GPU2DState&>(State2D_B) = GPU.GPU2D_B;
        UpdateBGVRAM(State2D_A);
        UpdateBGVRAM(State2D_B);

        switch ((dispcnt >> 16) & 0x3)
        {
        case 2: // VRAM display
            {
                u32 vrambank = (dispcnt >> 18) & 0x3;
                if (GPU.VRAMMap_LCDC & (1<<vrambank))
                    displayLine = &((u16*)GPU.VRAM[vrambank])[vcount * 256];
            }
            break;

        case 3: // FIFO display
            displayLine = GPU.DispFIFOBuffer;
            break;
        }
    }

    RenderScanline(vcount, dstA, dstB, GPU.ScreensEnabled, displayLine, GPU.MasterBrightnessA, GPU.MasterBrightnessB);

    // perform display capture if enabled
    if (vcount < 192)
    {
        u32 captureCnt = GPU.CaptureCnt;
        if (GPU.CaptureEnable && CaptureNeeded(vcount, captureCnt, GPU.VRAMMap_LCDC))
        {
            const u32* srcA = (captureCnt & (1<<24)) ? Output3D : Output2D[0];
            WriteCapture(vcount, captureCnt, srcA, CaptureSourceB(vcount, captureCnt, dispcnt));
        }
    }
}

void SoftRenderer::DrawSprites(u32 line)
{
    if (Threaded2D)
    {
        Job2D& job = BeginJob();
        job.Sprites = true;
        job.Line = line;
        job.Capture = false;

        BeginJobState(job);
        UpdateOBJVRAM(GPU.GPU2D_A);
        UpdateOBJVRAM(GPU.GPU2D_B);

        SubmitJob();
        return;
    }

    static_cast<GPU2DState&>(State2D_A) = GPU.GPU2D_A;
    static_cast<GPU2DState&>(State2D_B) = GPU.GPU2D_B;
    UpdateOBJVRAM(State2D_A);
    UpdateOBJVRAM(State2D_B);

    Rend2D_A->DrawSprites(line);
    Rend2D_B->DrawSprites(line);
}

void SoftRenderer::RenderScanline(u32 line, u32* dstA, u32* dstB, bool screensEnabled,
                                  const u16* displayLine, u16 brightnessA, u16 brightnessB)
{
    if (line < 192)
    {
        // draw BG/OBJ layers
        Rend2D_A->DrawScanline(line);
        Rend2D_B->DrawScanline(line);

        // draw the final screen output
        DrawScanlineA(line, dstA, displayLine, brightnessA);
        DrawScanlineB(line, dstB, brightnessB);
    }
    else
    {
        // if scanlines outside VCOUNT range 0..191 were to be visible, fill them white
        // this may happen if VCOUNT is written to during active display
        // the actual hardware behavior depends on the screen model, and suggests that
        // no video signal is output for such scanlines

        for (int i = 0; i < 256; i++)
        {
            dstA[i] = 0x3F3F3F;
            dstB[i] = 0x3F3F3F;
        }
    }

    if (screensEnabled)
    {
        // expand the color from 6-bit to 8-bit
        ExpandColor(dstA);
        ExpandColor(dstB);
    }
    else
    {
        // if the screens are disabled: fill the framebuffer black
        for (int i = 0; i < 256; i++)
        {
            dstA[i] = 0xFF000000;
            dstB[i] = 0xFF000000;
        }
    }
}

void SoftRenderer::DrawScanlineA(u32 line, u32* dst, const u16* displayLine, u16 brightness)
{
    u32 dispcnt = State2D_A.DispCnt;
    switch ((dispcnt >> 16) & 0x3)
    {
    case 0: // screen off
        {
            for (int i = 0; i < 256; i++)
                dst[i] = 0x3F3F3F;
        }
        return;

    case 1: // regular display
        {
            for (int i = 0; i < 256; i+=2)
                *(u64*)&dst[i] = *(u64*)&Output2D[0][i];
        }
        break;

    case 2: // VRAM display
        {
            // displayLine is only set if the selected bank is mapped to LCDC
            if (displayLine)
            {
                for (int i = 0; i < 256; i++)
                {
                    u16 color = displayLine[i];
                    u8 r = (color & 0x001F) << 1;
                    u8 g = (color & 0x03E0) >> 4;
                    u8 b = (color & 0x7C00) >> 9;

                    dst[i] = r | (g << 8) | (b << 16);
                }
            }
            else
            {
                for (int i = 0; i < 256; i++)
                    dst[i] = 0;
            }
        }
        break;

    case 3: // FIFO display
        {
            for (int i = 0; i < 256; i++)
            {
                u16 color = displayLine[i];
                u8 r = (color & 0x001F) << 1;
                u8 g = (color & 0x03E0) >> 4;
                u8 b = (color & 0x7C00) >> 9;

                dst[i] = r | (g << 8) | (b << 16);
            }
        }
        break;
    }

    ApplyMasterBrightness(brightness, dst);
}

void SoftRenderer::DrawScanlineB(u32 line, u32* dst, u16 brightness)
{
    u32 dispcnt = State2D_B.DispCnt;
    switch ((dispcnt >> 16) & 0x1)
    {
    case 0: // screen off
        {
            for (int i = 0; i < 256; i++)
                dst[i] = 0xFF3F3F3F;
        }
        return;

    case 1: // regular display
        {
            for (int i = 0; i < 256; i+=2)
                *(u64*)&dst[i] = *(u64*)&Output2D[1][i];
        }
        break;
    }

    ApplyMasterBrightness(brightness, dst);
}

void SoftRenderer::UpdateBGVRAM(const melonDS::GPU2D& unit)
{
    // the BG layers aren't drawn if the 2D engine is disabled or blanked
    if (!unit.Enabled || unit.ForcedBlank)
        return;

    if (unit.Num == 0)
    {
        auto bgDirty = GPU.VRAMDirty_ABG.DeriveState(GPU.VRAMMap_ABG, GPU);
        auto bgExtPalDirty = GPU.VRAMDirty_ABGExtPal.DeriveState(GPU.VRAMMap_ABGExtPal, GPU);
        auto objExtPalDirty = GPU.VRAMDirty_AOBJExtPal.DeriveState(&GPU.VRAMMap_AOBJExtPal, GPU);

        // queued scanlines still read the old contents
        if (bgDirty || bgExtPalDirty || objExtPalDirty)
            WaitIdle();

        GPU.MakeVRAMFlat_ABGCoherent(bgDirty);
        GPU.MakeVRAMFlat_ABGExtPalCoherent(bgExtPalDirty);
        GPU.MakeVRAMFlat_AOBJExtPalCoherent(objExtPalDirty);
    }
    else
    {
        auto bgDirty = GPU.VRAMDirty_BBG.DeriveState(GPU.VRAMMap_BBG, GPU);
        auto bgExtPalDirty = GPU.VRAMDirty_BBGExtPal.DeriveState(GPU.VRAMMap_BBGExtPal, GPU);
        auto objExtPalDirty = GPU.VRAMDirty_BOBJExtPal.DeriveState(&GPU.VRAMMap_BOBJExtPal, GPU);

        if (bgDirty || bgExtPalDirty || objExtPalDirty)
            WaitIdle();

        GPU.MakeVRAMFlat_BBGCoherent(bgDirty);
        GPU.MakeVRAMFlat_BBGExtPalCoherent(bgExtPalDirty);
        GPU.MakeVRAMFlat_BOBJExtPalCoherent(objExtPalDirty);
    }
}

void SoftRenderer::UpdateOBJVRAM(const melonDS::GPU2D& unit)
{
    // the OBJ buffers don't get updated at all if the 2D engine is disabled
    if (!unit.Enabled)
        return;

    if (unit.Num == 0)
    {
        auto objDirty = GPU.VRAMDirty_AOBJ.DeriveState(GPU.VRAMMap_AOBJ, GPU);
        if (objDirty)
            WaitIdle();
        GPU.MakeVRAMFlat_AOBJCoherent(objDirty);
    }
    else
    {
        auto objDirty = GPU.VRAMDirty_BOBJ.DeriveState(GPU.VRAMMap_BOBJ, GPU);
        if (objDirty)
            WaitIdle();
        GPU.MakeVRAMFlat_BOBJCoherent(objDirty);
    }
}

bool SoftRenderer::CaptureNeeded(u32 line, u32 captureCnt, u32 lcdcMap) const
{
    u32 height;
    u32 sz = (captureCnt >> 20) & 0x3;
    if (sz == 0)
        height = 128;
    else
        height = 64 * sz;

    if (line >= height)
        return false;

    u32 dstvram = (captureCnt >> 16) & 0x3;
    return (lcdcMap & (1<<dstvram)) != 0;
}

const u16* SoftRenderer::CaptureSourceB(u32 line, u32 captureCnt, u32 dispCnt) const
{
    if (captureCnt & (1<<25))
        return GPU.DispFIFOBuffer;

    u32 srcvram = (dispCnt >> 18) & 0x3;
    if (!(GPU.VRAMMap_LCDC & (1<<srcvram)))
        return nullptr;

    u32 offset = line * 256;
    if (((dispCnt >> 16) & 0x3) != 2)
        offset += (((captureCnt >> 26) & 0x3) << 14);

    return &((const u16*)GPU.VRAM[srcvram])[offset & 0xFFFF];
}

void SoftRenderer::WriteCapture(u32 line, u32 captureCnt, const u32* srcA, const u16* srcB)
{
    u32 width;
    u32 sz = (captureCnt >> 20) & 0x3;
    if (sz == 0)
        width = 128;
    else
        width = 256;

    u32 dstvram = (captureCnt >> 16) & 0x3;
    u16* dst = (u16*)GPU.VRAM[dstvram];
    u32 dstaddr = (((captureCnt >> 18) & 0x3) << 14) + (line * width);
    dst += (dstaddr & 0xFFFF);

    static_assert(VRAMDirtyGranularity == 512);
    GPU.VRAMDirty[dstvram][(dstaddr * 2) / VRAMDirtyGranularity] = true;

    switch ((captureCnt >> 29) & 0x3)
    {
    case 0: // source A
        {
            for (u32 i = 0; i < width; i++)
            {
                u32 val = srcA[i];

                u32 r = (val >> 1) & 0x1F;
                u32 g = (val >> 9) & 0x1F;
                u32 b = (val >> 17) & 0x1F;
                u32 a = ((val >> 24) != 0) ? 0x8000 : 0;

                dst[i] = r | (g << 5) | (b << 10) | a;
            }
        }
        break;

    case 1: // source B
        {
            if (srcB)
            {
                for (u32 i = 0; i < width; i++)
                    dst[i] = srcB[i];
            }
            else
            {
                for (u32 i = 0; i < width; i++)
                    dst[i] = 0;
            }
        }
        break;

    case 2: // sources A+B
    case 3:
        {
            u32 eva = captureCnt & 0x1F;
            u32 evb = (captureCnt >> 8) & 0x1F;

            // checkme
            if (eva > 16) eva = 16;
            if (evb > 16) evb = 16;

            if (srcB)
            {
                for (u32 i = 0; i < width; i++)
                {
                    u32 val = srcA[i];

                    u32 rA = (val >> 1) & 0x1F;
                    u32 gA = (val >> 9) & 0x1F;
                    u32 bA = (val >> 17) & 0x1F;
                    u32 aA = ((val >> 24) != 0) ? 1 : 0;

                    val = srcB[i];

                    u32 rB = val & 0x1F;
                    u32 gB = (val >> 5) & 0x1F;
                    u32 bB = (val >> 10) & 0x1F;
                    u32 aB = val >> 15;

                    u32 rD = ((rA * aA * eva) + (rB * aB * evb) + 8) >> 4;
                    u32 gD = ((gA * aA * eva) + (gB * aB * evb) + 8) >> 4;
                    u32 bD = ((bA * aA * eva) + (bB * aB * evb) + 8) >> 4;
                    u32 aD = (eva>0 ? aA : 0) | (evb>0 ? aB : 0);

                    if (rD > 0x1F) rD = 0x1F;
                    if (gD > 0x1F) gD = 0x1F;
                    if (bD > 0x1F) bD = 0x1F;

                    dst[i] = rD | (gD << 5) | (bD << 10) | (aD << 15);
                }
            }
            else
            {
                for (u32 i = 0; i < width; i++)
                {
                    u32 val = srcA[i];

                    u32 rA = (val >> 1) & 0x1F;
                    u32 gA = (val >> 9) & 0x1F;
                    u32 bA = (val >> 17) & 0x1F;
                    u32 aA = ((val >> 24) != 0) ? 1 : 0;

                    u32 rD = ((rA * aA * eva) + 8) >> 4;
                    u32 gD = ((gA * aA * eva) + 8) >> 4;
                    u32 bD = ((bA * aA * eva) + 8) >> 4;
                    u32 aD = (eva>0 ? aA : 0);

                    dst[i] = rD | (gD << 5) | (bD << 10) | (aD << 15);
                }
            }
        }
        break;
    }
}


void SoftRenderer::VBlank()
{
    // the capture ends here, the CPU may read the result from now on
    for (u32 bank = 0; bank < 4; bank++)
        SyncCaptures(bank);

    CaptureImmediate = false;
}

void SoftRenderer::SyncVRAMCapture(u32 bank, u32 start, u32 len, bool complete)
{
    // the CPU accesses a captured block
    SyncCaptures(bank);

    if (Threaded2D && GPU.CaptureEnable)
        CaptureImmediate = true;
}

void SoftRenderer::Start3DRendering()
{
    // queued scanlines may still read the 3D renderer's buffer, which the new frame overwrites;
    // rather than waiting for them, let them read a copy
    if (Threaded2D && JobsSubmitted.load(std::memory_order_relaxed) != JobsDone.load())
    {
        if (!Line3DSnapshot)
            Line3DSnapshot = std::make_unique<u32[]>(SoftRenderer3D::LineBufferSize);
        memcpy(Line3DSnapshot.get(), Soft3D->LineBuffer(), SoftRenderer3D::LineBufferSize * sizeof(u32));
        Line3DBase.store(Line3DSnapshot.get());

        // only the scanline in progress may have taken the old address
        WaitForJobs(JobsStarted.load());
    }

    Rend3D->RenderFrame();
}

void SoftRenderer::Restart3DRendering()
{
    WaitIdle();
    Rend3D->RestartFrame();
}

void SoftRenderer::SwapBuffers()
{
    // finish the frame before it is shown
    WaitIdle();
    Line3DBase.store(Soft3D->LineBuffer());

    BackBuffer ^= 1;
}


void SoftRenderer::SetThreaded2D(bool threaded)
{
    if (threaded == Threaded2D)
        return;

    if (threaded)
    {
        if (!Jobs)
        {
            Jobs = std::make_unique<Job2D[]>(NumJobs);
            PaletteCopies = std::make_unique<u8[]>(NumMemCopies * sizeof(GPU.Palette));
            OAMCopies = std::make_unique<u8[]>(NumMemCopies * sizeof(GPU.OAM));
        }

        JobsSubmitted = 0;
        JobsStarted = 0;
        JobsDone = 0;
        JobsRetired = 0;
        Line3DBase = Soft3D->LineBuffer();
        memset(CaptureJobs, 0, sizeof(CaptureJobs));
        CaptureImmediate = false;
        memset(PaletteCopyJobs, 0, sizeof(PaletteCopyJobs));
        memset(OAMCopyJobs, 0, sizeof(OAMCopyJobs));
        CurPaletteCopy = 0;
        CurOAMCopy = 0;

        // the first job takes fresh copies
        GPU.PaletteWritten = true;
        GPU.OAMWritten = true;

        StopThread2D = false;
        WorkerSleeping = false;
        EmuWaiting = false;
        Platform::Semaphore_Reset(Sema_Work);
        Platform::Semaphore_Reset(Sema_Done);

        Thread2D = Platform::Thread_Create([this]() { RenderThread2DFunc(); });
        if (!Thread2D)
            return; // no thread support, stay unthreaded

        Threaded2D = true;
    }
    else
    {
        WaitIdle();

        StopThread2D = true;
        Platform::Semaphore_Post(Sema_Work);
        Platform::Thread_Wait(Thread2D);
        Platform::Thread_Free(Thread2D);
        Thread2D = nullptr;

        Threaded2D = false;

        Soft2D_A->SetMemory(GPU.Palette, GPU.OAM);
        Soft2D_B->SetMemory(GPU.Palette, GPU.OAM);
    }
}

void SoftRenderer::RenderThread2DFunc()
{
    u32 done = JobsDone.load(std::memory_order_relaxed);

    for (;;)
    {
        if (JobsSubmitted.load(std::memory_order_acquire) == done)
        {
            auto ready = [&]() {
                return JobsSubmitted.load(std::memory_order_acquire) != done
                    || StopThread2D.load(std::memory_order_relaxed);
            };

            if (!SpinUntil(ready, RenderThreadSpin))
            {
                // the emulation thread posts Sema_Work if it sees this flag after submitting a job
                WorkerSleeping = true;
                if (JobsSubmitted.load() == done && !StopThread2D)
                    Platform::Semaphore_Wait(Sema_Work);
                WorkerSleeping = false;
            }

            if (StopThread2D)
                return;

            continue;
        }

        // see Start3DRendering
        JobsStarted.store(done + 1);
        RunJob(Jobs[done % NumJobs]);

        done++;
        JobsDone.store(done);

        // wake the emulation thread once the job it waits for is done
        if (EmuWaiting.load() && (s32)(done - EmuWaitTarget.load()) >= 0 && EmuWaiting.exchange(false))
            Platform::Semaphore_Post(Sema_Done);
    }
}

void SoftRenderer::RunJob(Job2D& job)
{
    if (!job.Sprites && job.VCount >= 192)
    {
        // no 2D state needed for these
        RenderScanline(job.VCount, job.DstA, job.DstB, job.ScreensEnabled, nullptr, 0, 0);
        return;
    }

    static_cast<GPU2DState&>(State2D_A) = job.State[0];
    static_cast<GPU2DState&>(State2D_B) = job.State[1];

    u8* palette = &PaletteCopies[job.PaletteCopy * sizeof(GPU.Palette)];
    u8* oam = &OAMCopies[job.OAMCopy * sizeof(GPU.OAM)];
    Soft2D_A->SetMemory(palette, oam);
    Soft2D_B->SetMemory(palette, oam);

    if (job.Sprites)
    {
        Rend2D_A->DrawSprites(job.Line);
        Rend2D_B->DrawSprites(job.Line);
        return;
    }

    if (job.Line3DCopied)
        Output3D = job.Line3D;
    else
        Output3D = const_cast<u32*>(Line3DBase.load() + job.Line3DOffset);
    RenderScanline(job.VCount, job.DstA, job.DstB, job.ScreensEnabled,
                   job.HasDisplayLine ? job.DisplayLine : nullptr,
                   job.BrightnessA, job.BrightnessB);

    if (job.Capture)
    {
        const u32* srcA = (job.CaptureCnt & (1<<24)) ? Output3D : Output2D[0];
        memcpy(job.CaptureA, srcA, sizeof(job.CaptureA));
    }
}

SoftRenderer::Job2D& SoftRenderer::BeginJob()
{
    u32 seq = JobsSubmitted.load(std::memory_order_relaxed);

    // the slot must be done and retired
    WaitForJobs(seq - NumJobs + 1);
    RetireJobs();

    return Jobs[seq % NumJobs];
}

void SoftRenderer::BeginJobState(Job2D& job)
{
    u32 seq = JobsSubmitted.load(std::memory_order_relaxed);

    // captures are written with the VRAM mapping of their scanline
    // if that changed, write them before anything gets to see the new mapping
    if (GPU.VRAMMap_LCDC != CaptureLCDCMap)
    {
        for (u32 bank = 0; bank < 4; bank++)
            SyncCaptures(bank);
        CaptureLCDCMap = GPU.VRAMMap_LCDC;
    }

    job.State[0] = GPU.GPU2D_A;
    job.State[1] = GPU.GPU2D_B;

    job.PaletteCopy = TakePaletteCopy(seq);
    job.OAMCopy = TakeOAMCopy(seq);
}

[[gnu::noinline]] u32 SoftRenderer::TakePaletteCopy(u32 seq)
{
    if (GPU.PaletteWritten)
    {
        u32 next = (CurPaletteCopy + 1) % NumMemCopies;
        WaitForJobs(PaletteCopyJobs[next]);
        memcpy(&PaletteCopies[next * sizeof(GPU.Palette)], GPU.Palette, sizeof(GPU.Palette));
        CurPaletteCopy = next;
        GPU.PaletteWritten = false;
    }
    PaletteCopyJobs[CurPaletteCopy] = seq + 1;
    return CurPaletteCopy;
}

[[gnu::noinline]] u32 SoftRenderer::TakeOAMCopy(u32 seq)
{
    if (GPU.OAMWritten)
    {
        u32 next = (CurOAMCopy + 1) % NumMemCopies;
        WaitForJobs(OAMCopyJobs[next]);
        memcpy(&OAMCopies[next * sizeof(GPU.OAM)], GPU.OAM, sizeof(GPU.OAM));
        CurOAMCopy = next;
        GPU.OAMWritten = false;
    }
    OAMCopyJobs[CurOAMCopy] = seq + 1;
    return CurOAMCopy;
}

void SoftRenderer::SubmitJob()
{
    JobsSubmitted.store(JobsSubmitted.load(std::memory_order_relaxed) + 1);

    if (WorkerSleeping.load() && WorkerSleeping.exchange(false))
        Platform::Semaphore_Post(Sema_Work);
}

void SoftRenderer::WaitForJobs(u32 count)
{
    auto reached = [&]() { return (s32)(JobsDone.load() - count) >= 0; };

    if (reached() || SpinUntil(reached, EmuThreadSpin))
        return;

    for (;;)
    {
        // the render thread posts Sema_Done if it sees this flag after finishing the job we wait for
        EmuWaitTarget = count;
        EmuWaiting = true;
        if (reached())
        {
            EmuWaiting = false;
            return;
        }

        Platform::Semaphore_Wait(Sema_Done);
        if (reached())
            return;
    }
}

void SoftRenderer::RetireJobs()
{
    u32 done = JobsDone.load(std::memory_order_acquire);
    while (JobsRetired != done)
    {
        Job2D& job = Jobs[JobsRetired % NumJobs];
        if (job.Capture)
            WriteCapture(job.VCount, job.CaptureCnt, job.CaptureA, job.HasCaptureB ? job.CaptureB : nullptr);

        JobsRetired++;
    }
}

void SoftRenderer::WaitIdle()
{
    if (!Threaded2D)
        return;

    WaitForJobs(JobsSubmitted.load(std::memory_order_relaxed));
    RetireJobs();
}

void SoftRenderer::SyncCaptures(u32 bank)
{
    if (!Threaded2D)
        return;

    if ((s32)(CaptureJobs[bank] - JobsRetired) > 0)
    {
        WaitForJobs(CaptureJobs[bank]);
        RetireJobs();
    }
}

void SoftRenderer::ApplyMasterBrightness(u16 regval, u32* dst)
{
    u16 mode = regval >> 14;
    if (mode == 1)
    {
        // up
        u32 factor = regval & 0x1F;
        if (factor > 16) factor = 16;

        for (int i = 0; i < 256; i++)
            dst[i] = ColorBrightnessUp(dst[i], factor, 0x0);
    }
    else if (mode == 2)
    {
        // down
        u32 factor = regval & 0x1F;
        if (factor > 16) factor = 16;

        for (int i = 0; i < 256; i++)
            dst[i] = ColorBrightnessDown(dst[i], factor, 0xF);
    }
}

void SoftRenderer::ExpandColor(u32* dst)
{
    // convert to 32-bit BGRA
    // note: 32-bit RGBA would be more straightforward, but
    // BGRA seems to be more compatible (Direct2D soft, cairo...)
    for (int i = 0; i < 256; i+=2)
    {
        u64 c = *(u64*)&dst[i];

        u64 r = (c << 18) & 0xFC000000FC0000;
        u64 g = (c << 2) & 0xFC000000FC00;
        u64 b = (c >> 14) & 0xFC000000FC;
        c = r | g | b;

        *(u64*)&dst[i] = c | ((c & 0x00C0C0C000C0C0C0) >> 6) | 0xFF000000FF000000;
    }
}


bool SoftRenderer::GetFramebuffers(void** top, void** bottom)
{
    int frontbuf = BackBuffer ^ 1;
    *top = Framebuffer[frontbuf][0];
    *bottom = Framebuffer[frontbuf][1];
    return true;
}

}
