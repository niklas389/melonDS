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

#ifndef GPU_SOFT_H
#define GPU_SOFT_H

#include <atomic>
#include <memory>

#include "GPU.h"
#include "Platform.h"
#include "GPU2D_Soft.h"
#include "GPU3D_Soft.h"

namespace melonDS
{

class SoftRenderer : public Renderer
{
public:
    explicit SoftRenderer(melonDS::NDS& nds);
    ~SoftRenderer() override;
    bool Init() override { return true; }
    void Reset() override;
    void Stop() override;

    void PreSavestate() override;
    void PostSavestate() override;

    void SetRenderSettings(RendererSettings& settings) override;

    void DrawScanline(u32 line) override;
    void DrawSprites(u32 line) override;

    void VBlank() override;
    void VBlankEnd() override {};

    void AllocCapture(u32 bank, u32 start, u32 len) override {};
    void SyncVRAMCapture(u32 bank, u32 start, u32 len, bool complete) override;

    bool GetFramebuffers(void** top, void** bottom) override;
    void SwapBuffers() override;

    void Start3DRendering() override;
    void Restart3DRendering() override;

private:
    friend class SoftRenderer2D;
    friend class SoftRenderer3D;

    // lets the queued scanlines read a copy of the 3D buffer before a new frame is drawn into it
    void StartFrame3D();
    // whether VBlank() started the next 3D frame
    bool Early3DFrame = false;

    // The 2D renderers read these copies of the 2D engines' state.
    // Unthreaded, they are refreshed right before drawing;
    // threaded, the render thread loads them from each job.
    melonDS::GPU2D State2D_A;
    melonDS::GPU2D State2D_B;
    SoftRenderer2D* Soft2D_A;
    SoftRenderer2D* Soft2D_B;
    SoftRenderer3D* Soft3D;

    u32* Framebuffer[2][2];

    u32* Output3D;
    alignas(8) u32 Output2D[2][256];

    void RenderScanline(u32 line, u32* dstA, u32* dstB, bool screensEnabled,
                        const u16* displayLine, u16 brightnessA, u16 brightnessB);
    void DrawScanlineA(u32 line, u32* dst, const u16* displayLine, u16 brightness);
    void DrawScanlineB(u32 line, u32* dst, u16 brightness);

    void UpdateBGVRAM(const melonDS::GPU2D& unit);
    void UpdateOBJVRAM(const melonDS::GPU2D& unit);

    bool CaptureNeeded(u32 line, u32 captureCnt, u32 lcdcMap) const;
    const u16* CaptureSourceB(u32 line, u32 captureCnt, u32 dispCnt) const;
    void WriteCapture(u32 line, u32 captureCnt, const u32* srcA, const u16* srcB);

    void ApplyMasterBrightness(u16 regval, u32* dst);
    void ExpandColor(u32* dst);

    /*
        Threaded 2D rendering

        At each DrawScanline/DrawSprites call, the emulation thread queues a job holding everything
        the 2D renderers read that may change before the render thread gets to it: the 2D engine
        state, the 3D scanline, the VRAM or FIFO line for display and capture, and the palette and OAM
        (as references to copies that are only taken when they changed).
        The flat VRAM copies are shared; when they need updating, the emulation thread first waits
        until no queued job reads them any more.
        Display captures are written to VRAM by the emulation thread once their job is done, and at
        the latest when the capture ends at VBlank or the CPU accesses a captured VRAM block.
        The frame is complete when the buffers are swapped.
    */

    struct Job2D
    {
        bool Sprites;           // DrawSprites, otherwise DrawScanline
        bool ScreensEnabled;
        bool Capture;
        bool HasDisplayLine;
        bool HasCaptureB;
        u32 Line;
        u32 VCount;
        u32* DstA;
        u32* DstB;
        u16 BrightnessA;
        u16 BrightnessB;
        u32 CaptureCnt;
        u32 PaletteCopy;
        u32 OAMCopy;
        GPU2DState State[2];

        // the 3D scanline: at this offset in the 3D line buffer (see Line3DBase), or copied to Line3D
        bool Line3DCopied;
        u32 Line3DOffset;
        alignas(8) u32 Line3D[256];
        alignas(8) u16 DisplayLine[256];    // VRAM or FIFO display source for engine A
        alignas(8) u16 CaptureB[256];       // capture source B
        alignas(8) u32 CaptureA[256];       // capture source A, filled in by the render thread
    };

    static constexpr u32 NumJobs = 512;
    // enough for a new palette or OAM copy on every scanline of a frame
    static constexpr u32 NumMemCopies = 256;

    bool Threaded2D = false;
    Platform::Thread* Thread2D = nullptr;
    Platform::Semaphore* Sema_Work;
    Platform::Semaphore* Sema_Done;
    std::atomic<bool> StopThread2D {false};
    std::atomic<bool> WorkerSleeping {false};
    std::atomic<bool> EmuWaiting {false};
    std::atomic<u32> EmuWaitTarget {0};

    // job counts: submitted by the emulation thread, done by the render thread,
    // retired (captures written) by the emulation thread
    std::atomic<u32> JobsSubmitted {0};
    std::atomic<u32> JobsStarted {0};
    std::atomic<u32> JobsDone {0};

    // where queued scanlines read their 3D lines: the 3D renderer's buffer, or a copy of it taken
    // when the next 3D frame starts while scanlines of this one are still queued
    std::atomic<const u32*> Line3DBase {nullptr};
    std::unique_ptr<u32[]> Line3DSnapshot;
    u32 JobsRetired = 0;

    // job count that must be reached before the captures to each VRAM bank are all written
    u32 CaptureJobs[4] {};
    u32 CaptureLCDCMap = 0;
    // once the CPU has accessed the capture in progress, GPU stops telling us about further accesses,
    // so the rest of its scanlines are written right away
    bool CaptureImmediate = false;

    u32 CurPaletteCopy = 0;
    u32 CurOAMCopy = 0;
    u32 PaletteCopyJobs[NumMemCopies] {};
    u32 OAMCopyJobs[NumMemCopies] {};
    std::unique_ptr<u8[]> PaletteCopies;
    std::unique_ptr<u8[]> OAMCopies;
    std::unique_ptr<Job2D[]> Jobs;

    void SetThreaded2D(bool threaded);
    void RenderThread2DFunc();
    void RunJob(Job2D& job);

    Job2D& BeginJob();
    void SubmitJob();
    void BeginJobState(Job2D& job);
    u32 TakePaletteCopy(u32 seq);
    u32 TakeOAMCopy(u32 seq);
    // inlined into their callers, so that profiles show where the emulation thread waited
    [[gnu::always_inline]] inline void WaitForJobs(u32 count);
    void RetireJobs();
    [[gnu::always_inline]] inline void WaitIdle();
    [[gnu::always_inline]] inline void SyncCaptures(u32 bank);
};

}

#endif // GPU_SOFT_H
