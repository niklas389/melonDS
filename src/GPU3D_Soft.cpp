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

#include "GPU3D_Soft.h"

#include <algorithm>
#include <type_traits>
#include <stdio.h>
#include <string.h>
#include "NDS.h"
#include "GPU.h"

namespace melonDS
{

void RenderThreadFunc();


void SoftRenderer3D::StopRenderThread()
{
    if (RenderThreadRunning.load(std::memory_order_relaxed))
    {
        // Tell the render thread to stop drawing new frames, and finish up the current one.
        RenderThreadRunning = false;

        Platform::Semaphore_Post(Sema_RenderStart);

        Platform::Thread_Wait(RenderThread);
        Platform::Thread_Free(RenderThread);
        RenderThread = nullptr;
    }

    StopSplitThread();
}

void SoftRenderer3D::StartSplitThread()
{
    if (SplitThreadRunning.load(std::memory_order_relaxed))
        return;

    Platform::Semaphore_Reset(Sema_SplitStart);
    Platform::Semaphore_Reset(Sema_SplitDone);
    SplitThreadRunning = true;
    SplitThread = Platform::Thread_Create([this]() { SplitThreadFunc(); });
    if (!SplitThread)
        SplitThreadRunning = false;
}

void SoftRenderer3D::StopSplitThread()
{
    // only called while the render thread isn't drawing a frame
    if (!SplitThreadRunning.load(std::memory_order_relaxed))
        return;

    SplitThreadRunning = false;
    Platform::Semaphore_Post(Sema_SplitStart);
    Platform::Thread_Wait(SplitThread);
    Platform::Thread_Free(SplitThread);
    SplitThread = nullptr;
}

void SoftRenderer3D::SetSplit(bool split) noexcept
{
    if (Split != split)
    {
        Split = split;
        SetupRenderThread();
        EnableRenderThread();
    }
}

void SoftRenderer3D::SetupRenderThread()
{
    if (Threaded)
    {
        if (!RenderThreadRunning.load(std::memory_order_relaxed))
        { // If the render thread isn't already running...
            RenderThreadRunning = true; // "Time for work, render thread!"
            RenderThread = Platform::Thread_Create([this]() {
                RenderThreadFunc();
            });
        }

        // "Be on standby, but don't start rendering until I tell you to!"
        Platform::Semaphore_Reset(Sema_RenderStart);

        // "Oh, sorry, were you already in the middle of a frame from the last iteration?"
        if (RenderThreadRendering)
            // "Tell me when you're done, I'll wait here."
            Platform::Semaphore_Wait(Sema_RenderDone);

        // "All good? Okay, let me give you your training."
        // "(Maybe you're still the same thread, but I have to tell you this stuff anyway.)"

        // "This is the signal you'll send when you're done with a frame."
        // "I'll listen for it when I need to show something to the frontend."
        Platform::Semaphore_Reset(Sema_RenderDone);

        // "This is the signal I'll send when I want you to start rendering."
        // "Don't do anything until you get the message."
        Platform::Semaphore_Reset(Sema_RenderStart);

        // "This is the signal you'll send every time you finish drawing a line."
        // "I might need some of your scanlines before you finish the whole buffer,"
        // "so let me know as soon as you're done with each one."
        Platform::Semaphore_Reset(Sema_ScanlineCount);

        // the render thread is idle now, so the split thread can come or go
        if (Split)
            StartSplitThread();
        else
            StopSplitThread();
    }
    else
    {
        StopRenderThread();
    }
}

void SoftRenderer3D::EnableRenderThread()
{
    if (Threaded && Sema_RenderStart)
    {
        Platform::Semaphore_Post(Sema_RenderStart);
    }
}

SoftRenderer3D::SoftRenderer3D(melonDS::GPU3D& gpu3D, SoftRenderer& parent) noexcept
    : Renderer3D(gpu3D), Parent(parent)
{
    Sema_RenderStart = Platform::Semaphore_Create();
    Sema_RenderDone = Platform::Semaphore_Create();
    Sema_ScanlineCount = Platform::Semaphore_Create();
    Sema_SplitStart = Platform::Semaphore_Create();
    Sema_SplitDone = Platform::Semaphore_Create();
    Sema_TopHalfDone = Platform::Semaphore_Create();

    RenderThreadRunning = false;
    RenderThreadRendering = false;
    RenderThread = nullptr;
}

SoftRenderer3D::~SoftRenderer3D()
{
    StopRenderThread();

    Platform::Semaphore_Free(Sema_RenderStart);
    Platform::Semaphore_Free(Sema_RenderDone);
    Platform::Semaphore_Free(Sema_ScanlineCount);
    Platform::Semaphore_Free(Sema_SplitStart);
    Platform::Semaphore_Free(Sema_SplitDone);
    Platform::Semaphore_Free(Sema_TopHalfDone);
}

void SoftRenderer3D::Reset()
{
    // wait for a frame that's still being drawn before clearing what it uses
    SetupRenderThread();

    memset(ColorBuffer, 0, BufferSize * 2 * 4);
    memset(DepthBuffer, 0, BufferSize * 2 * 4);
    memset(AttrBuffer, 0, BufferSize * 2 * 4);

    PrevIsShadowMask = false;

    EnableRenderThread();
}

void SoftRenderer3D::SetThreaded(bool threaded) noexcept
{
    if (Threaded != threaded)
    {
        Threaded = threaded;
        SetupRenderThread();
        EnableRenderThread();
    }
}

void SoftRenderer3D::TextureLookup(u32 texparam, u32 texpal, s16 s, s16 t, u16* color, u8* alpha) const
{
    // TODO: consider using texture cache
    // however, I like the idea of having a "hardware accurate" path

    u32 vramaddr = (texparam & 0xFFFF) << 3;

    s32 width = 8 << ((texparam >> 20) & 0x7);
    s32 height = 8 << ((texparam >> 23) & 0x7);

    s >>= 4;
    t >>= 4;

    // texture wrapping
    // TODO: optimize this somehow
    // testing shows that it's hardly worth optimizing, actually

    if (texparam & (1<<16))
    {
        if (texparam & (1<<18))
        {
            if (s & width) s = (width-1) - (s & (width-1));
            else           s = (s & (width-1));
        }
        else
            s &= width-1;
    }
    else
    {
        if (s < 0) s = 0;
        else if (s >= width) s = width-1;
    }

    if (texparam & (1<<17))
    {
        if (texparam & (1<<19))
        {
            if (t & height) t = (height-1) - (t & (height-1));
            else            t = (t & (height-1));
        }
        else
            t &= height-1;
    }
    else
    {
        if (t < 0) t = 0;
        else if (t >= height) t = height-1;
    }

    u8 alpha0;
    if (texparam & (1<<29)) alpha0 = 0;
    else                    alpha0 = 31;

    switch ((texparam >> 26) & 0x7)
    {
    case 1: // A3I5
        {
            vramaddr += ((t * width) + s);
            u8 pixel = GPU.ReadVRAMFlat_Texture<u8>(vramaddr);

            texpal <<= 4;
            *color = GPU.ReadVRAMFlat_TexPal<u16>(texpal + ((pixel&0x1F)<<1));
            *alpha = ((pixel >> 3) & 0x1C) + (pixel >> 6);
        }
        break;

    case 2: // 4-color
        {
            vramaddr += (((t * width) + s) >> 2);
            u8 pixel = GPU.ReadVRAMFlat_Texture<u8>(vramaddr);
            pixel >>= ((s & 0x3) << 1);
            pixel &= 0x3;

            texpal <<= 3;
            *color = GPU.ReadVRAMFlat_TexPal<u16>(texpal + (pixel<<1));
            *alpha = (pixel==0) ? alpha0 : 31;
        }
        break;

    case 3: // 16-color
        {
            vramaddr += (((t * width) + s) >> 1);
            u8 pixel = GPU.ReadVRAMFlat_Texture<u8>(vramaddr);
            if (s & 0x1) pixel >>= 4;
            else         pixel &= 0xF;

            texpal <<= 4;
            *color = GPU.ReadVRAMFlat_TexPal<u16>(texpal + (pixel<<1));
            *alpha = (pixel==0) ? alpha0 : 31;
        }
        break;

    case 4: // 256-color
        {
            vramaddr += ((t * width) + s);
            u8 pixel = GPU.ReadVRAMFlat_Texture<u8>(vramaddr);

            texpal <<= 4;
            *color = GPU.ReadVRAMFlat_TexPal<u16>(texpal + (pixel<<1));
            *alpha = (pixel==0) ? alpha0 : 31;
        }
        break;

    case 5: // compressed
        {
            vramaddr += ((t & 0x3FC) * (width>>2)) + (s & 0x3FC);
            vramaddr += (t & 0x3);
            vramaddr &= 0x7FFFF; // address used for all calcs wraps around after slot 3

            u32 slot1addr = 0x20000 + ((vramaddr & 0x1FFFC) >> 1);
            if (vramaddr >= 0x40000)
                slot1addr += 0x10000;

            u8 val;
            if (vramaddr >= 0x20000 && vramaddr < 0x40000) // reading slot 1 for texels should always read 0
                val = 0;
            else
            {
                val = GPU.ReadVRAMFlat_Texture<u8>(vramaddr);
                val >>= (2 * (s & 0x3));
            }

            u16 palinfo = GPU.ReadVRAMFlat_Texture<u16>(slot1addr);
            u32 paloffset = (palinfo & 0x3FFF) << 2;
            texpal <<= 4;

            switch (val & 0x3)
            {
            case 0:
                *color = GPU.ReadVRAMFlat_TexPal<u16>(texpal + paloffset);
                *alpha = 31;
                break;

            case 1:
                *color = GPU.ReadVRAMFlat_TexPal<u16>(texpal + paloffset + 2);
                *alpha = 31;
                break;

            case 2:
                if ((palinfo >> 14) == 1)
                {
                    u16 color0 = GPU.ReadVRAMFlat_TexPal<u16>(texpal + paloffset);
                    u16 color1 = GPU.ReadVRAMFlat_TexPal<u16>(texpal + paloffset + 2);

                    u32 r0 = color0 & 0x001F;
                    u32 g0 = color0 & 0x03E0;
                    u32 b0 = color0 & 0x7C00;
                    u32 r1 = color1 & 0x001F;
                    u32 g1 = color1 & 0x03E0;
                    u32 b1 = color1 & 0x7C00;

                    u32 r = (r0 + r1) >> 1;
                    u32 g = ((g0 + g1) >> 1) & 0x03E0;
                    u32 b = ((b0 + b1) >> 1) & 0x7C00;

                    *color = r | g | b;
                }
                else if ((palinfo >> 14) == 3)
                {
                    u16 color0 = GPU.ReadVRAMFlat_TexPal<u16>(texpal + paloffset);
                    u16 color1 = GPU.ReadVRAMFlat_TexPal<u16>(texpal + paloffset + 2);

                    u32 r0 = color0 & 0x001F;
                    u32 g0 = color0 & 0x03E0;
                    u32 b0 = color0 & 0x7C00;
                    u32 r1 = color1 & 0x001F;
                    u32 g1 = color1 & 0x03E0;
                    u32 b1 = color1 & 0x7C00;

                    u32 r = (r0*5 + r1*3) >> 3;
                    u32 g = ((g0*5 + g1*3) >> 3) & 0x03E0;
                    u32 b = ((b0*5 + b1*3) >> 3) & 0x7C00;

                    *color = r | g | b;
                }
                else
                    *color = GPU.ReadVRAMFlat_TexPal<u16>(texpal + paloffset + 4);
                *alpha = 31;
                break;

            case 3:
                if ((palinfo >> 14) == 2)
                {
                    *color = GPU.ReadVRAMFlat_TexPal<u16>(texpal + paloffset + 6);
                    *alpha = 31;
                }
                else if ((palinfo >> 14) == 3)
                {
                    u16 color0 = GPU.ReadVRAMFlat_TexPal<u16>(texpal + paloffset);
                    u16 color1 = GPU.ReadVRAMFlat_TexPal<u16>(texpal + paloffset + 2);

                    u32 r0 = color0 & 0x001F;
                    u32 g0 = color0 & 0x03E0;
                    u32 b0 = color0 & 0x7C00;
                    u32 r1 = color1 & 0x001F;
                    u32 g1 = color1 & 0x03E0;
                    u32 b1 = color1 & 0x7C00;

                    u32 r = (r0*3 + r1*5) >> 3;
                    u32 g = ((g0*3 + g1*5) >> 3) & 0x03E0;
                    u32 b = ((b0*3 + b1*5) >> 3) & 0x7C00;

                    *color = r | g | b;
                    *alpha = 31;
                }
                else
                {
                    *color = 0;
                    *alpha = 0;
                }
                break;
            }
        }
        break;

    case 6: // A5I3
        {
            vramaddr += ((t * width) + s);
            u8 pixel = GPU.ReadVRAMFlat_Texture<u8>(vramaddr);

            texpal <<= 4;
            *color = GPU.ReadVRAMFlat_TexPal<u16>(texpal + ((pixel&0x7)<<1));
            *alpha = (pixel >> 3);
        }
        break;

    case 7: // direct color
        {
            vramaddr += (((t * width) + s) << 1);
            *color = GPU.ReadVRAMFlat_Texture<u16>(vramaddr);
            *alpha = (*color & 0x8000) ? 31 : 0;
        }
        break;
    }
}

// depth test is 'less or equal' instead of 'less than' under the following conditions:
// * when drawing a front-facing pixel over an opaque back-facing pixel
// * when drawing wireframe edges, under certain conditions (TODO)
//
// range is different based on depth-buffering mode
// Z-buffering: +-0x200
// W-buffering: +-0xFF

[[gnu::always_inline]] static inline bool DepthTest_Equal_Z(s32 dstz, s32 z, u32 dstattr)
{
    s32 diff = dstz - z;
    if ((u32)(diff + 0x200) <= 0x400)
        return true;

    return false;
}

[[gnu::always_inline]] static inline bool DepthTest_Equal_W(s32 dstz, s32 z, u32 dstattr)
{
    s32 diff = dstz - z;
    if ((u32)(diff + 0xFF) <= 0x1FE)
        return true;

    return false;
}

[[gnu::always_inline]] static inline bool DepthTest_LessThan(s32 dstz, s32 z, u32 dstattr)
{
    if (z < dstz)
        return true;

    return false;
}

[[gnu::always_inline]] static inline bool DepthTest_LessThan_FrontFacing(s32 dstz, s32 z, u32 dstattr)
{
    if ((dstattr & 0x00400010) == 0x00000010) // opaque, back facing
    {
        if (z <= dstz)
            return true;
    }
    else
    {
        if (z < dstz)
            return true;
    }

    return false;
}

u32 SoftRenderer3D::AlphaBlend(u32 srccolor, u32 dstcolor, u32 alpha) const noexcept
{
    u32 dstalpha = dstcolor >> 24;

    if (dstalpha == 0)
        return srccolor;

    u32 srcR = srccolor & 0x3F;
    u32 srcG = (srccolor >> 8) & 0x3F;
    u32 srcB = (srccolor >> 16) & 0x3F;

    if (GPU3D.RenderDispCnt & (1<<3))
    {
        u32 dstR = dstcolor & 0x3F;
        u32 dstG = (dstcolor >> 8) & 0x3F;
        u32 dstB = (dstcolor >> 16) & 0x3F;

        alpha++;
        srcR = ((srcR * alpha) + (dstR * (32-alpha))) >> 5;
        srcG = ((srcG * alpha) + (dstG * (32-alpha))) >> 5;
        srcB = ((srcB * alpha) + (dstB * (32-alpha))) >> 5;
        alpha--;
    }

    if (alpha > dstalpha)
        dstalpha = alpha;

    return srcR | (srcG << 8) | (srcB << 16) | (dstalpha << 24);
}

u32 SoftRenderer3D::RenderPixel(const PixelShading& shading, u8 vr, u8 vg, u8 vb, s16 s, s16 t) const
{
    u8 r, g, b, a;

    u32 blendmode = shading.BlendMode;
    u32 polyalpha = shading.PolyAlpha;
    bool wireframe = shading.Wireframe;

    if (blendmode == 2)
    {
        if (shading.Highlight)
        {
            // highlight mode: color is calculated normally
            // except all vertex color components are set
            // to the red component
            // the toon color is added to the final color

            vg = vr;
            vb = vr;
        }
        else
        {
            // toon mode: vertex color is replaced by toon color

            u16 tooncolor = GPU3D.RenderToonTable[vr >> 1];

            vr = (tooncolor << 1) & 0x3E; if (vr) vr++;
            vg = (tooncolor >> 4) & 0x3E; if (vg) vg++;
            vb = (tooncolor >> 9) & 0x3E; if (vb) vb++;
        }
    }

    if (shading.Textured)
    {
        u8 tr, tg, tb;

        u16 tcolor; u8 talpha;
        TextureLookup(shading.TexParam, shading.TexPalette, s, t, &tcolor, &talpha);

        tr = (tcolor << 1) & 0x3E; if (tr) tr++;
        tg = (tcolor >> 4) & 0x3E; if (tg) tg++;
        tb = (tcolor >> 9) & 0x3E; if (tb) tb++;

        if (blendmode & 0x1)
        {
            // decal

            if (talpha == 0)
            {
                r = vr;
                g = vg;
                b = vb;
            }
            else if (talpha == 31)
            {
                r = tr;
                g = tg;
                b = tb;
            }
            else
            {
                r = ((tr * talpha) + (vr * (31-talpha))) >> 5;
                g = ((tg * talpha) + (vg * (31-talpha))) >> 5;
                b = ((tb * talpha) + (vb * (31-talpha))) >> 5;
            }
            a = polyalpha;
        }
        else
        {
            // modulate

            r = ((tr+1) * (vr+1) - 1) >> 6;
            g = ((tg+1) * (vg+1) - 1) >> 6;
            b = ((tb+1) * (vb+1) - 1) >> 6;
            a = ((talpha+1) * (polyalpha+1) - 1) >> 5;
        }
    }
    else
    {
        r = vr;
        g = vg;
        b = vb;
        a = polyalpha;
    }

    if ((blendmode == 2) && shading.Highlight)
    {
        u16 tooncolor = GPU3D.RenderToonTable[vr >> 1];

        vr = (tooncolor << 1) & 0x3E; if (vr) vr++;
        vg = (tooncolor >> 4) & 0x3E; if (vg) vg++;
        vb = (tooncolor >> 9) & 0x3E; if (vb) vb++;

        r += vr;
        g += vg;
        b += vb;

        if (r > 63) r = 63;
        if (g > 63) g = 63;
        if (b > 63) b = 63;
    }

    // checkme: can wireframe polygons use texture alpha?
    if (wireframe) a = 31;

    return r | (g << 8) | (b << 16) | (a << 24);
}

void SoftRenderer3D::PlotTranslucentPixel(u32 pixeladdr, u32 color, u32 z, u32 polyattr, u32 shadow)
{
    u32 dstattr = AttrBuffer[pixeladdr];
    u32 attr = (polyattr & 0xE0F0) | ((polyattr >> 8) & 0xFF0000) | (1<<22) | (dstattr & 0xFF001F0F);

    if (shadow)
    {
        // for shadows, opaque pixels are also checked
        if (dstattr & (1<<22))
        {
            if ((dstattr & 0x007F0000) == (attr & 0x007F0000))
                return;
        }
        else
        {
            if ((dstattr & 0x3F000000) == (polyattr & 0x3F000000))
                return;
        }
    }
    else
    {
        // skip if translucent polygon IDs are equal
        if ((dstattr & 0x007F0000) == (attr & 0x007F0000))
            return;
    }

    // fog flag
    if (!(dstattr & (1<<15)))
        attr &= ~(1<<15);

    color = AlphaBlend(color, ColorBuffer[pixeladdr], color>>24);

    if (z != -1)
        DepthBuffer[pixeladdr] = z;

    ColorBuffer[pixeladdr] = color;
    AttrBuffer[pixeladdr] = attr;
}

void SoftRenderer3D::SetupPolygonLeftEdge(SoftRenderer3D::RendererPolygon* rp, s32 y) const
{
    Polygon* polygon = rp->PolyData;

    while (y >= polygon->Vertices[rp->NextVL]->FinalPosition[1] && rp->CurVL != polygon->VBottom)
    {
        rp->CurVL = rp->NextVL;

        if (polygon->FacingView)
        {
            rp->NextVL = rp->CurVL + 1;
            if (rp->NextVL >= polygon->NumVertices)
                rp->NextVL = 0;
        }
        else
        {
            rp->NextVL = rp->CurVL - 1;
            if ((s32)rp->NextVL < 0)
                rp->NextVL = polygon->NumVertices - 1;
        }
    }

    rp->XL = rp->SlopeL.Setup(polygon->Vertices[rp->CurVL]->FinalPosition[0], polygon->Vertices[rp->NextVL]->FinalPosition[0],
                              polygon->Vertices[rp->CurVL]->FinalPosition[1], polygon->Vertices[rp->NextVL]->FinalPosition[1],
                              polygon->FinalW[rp->CurVL], polygon->FinalW[rp->NextVL], y, polygon->WBuffer);
}

void SoftRenderer3D::SetupPolygonRightEdge(SoftRenderer3D::RendererPolygon* rp, s32 y) const
{
    Polygon* polygon = rp->PolyData;

    while (y >= polygon->Vertices[rp->NextVR]->FinalPosition[1] && rp->CurVR != polygon->VBottom)
    {
        rp->CurVR = rp->NextVR;

        if (polygon->FacingView)
        {
            rp->NextVR = rp->CurVR - 1;
            if ((s32)rp->NextVR < 0)
                rp->NextVR = polygon->NumVertices - 1;
        }
        else
        {
            rp->NextVR = rp->CurVR + 1;
            if (rp->NextVR >= polygon->NumVertices)
                rp->NextVR = 0;
        }
    }

    rp->XR = rp->SlopeR.Setup(polygon->Vertices[rp->CurVR]->FinalPosition[0], polygon->Vertices[rp->NextVR]->FinalPosition[0],
                              polygon->Vertices[rp->CurVR]->FinalPosition[1], polygon->Vertices[rp->NextVR]->FinalPosition[1],
                              polygon->FinalW[rp->CurVR], polygon->FinalW[rp->NextVR], y, polygon->WBuffer);
}

void SoftRenderer3D::SetupPolygon(SoftRenderer3D::RendererPolygon* rp, Polygon* polygon) const
{
    u32 nverts = polygon->NumVertices;

    u32 vtop = polygon->VTop, vbot = polygon->VBottom;
    s32 ytop = polygon->YTop, ybot = polygon->YBottom;

    rp->PolyData = polygon;

    rp->CurVL = vtop;
    rp->CurVR = vtop;

    if (polygon->FacingView)
    {
        rp->NextVL = rp->CurVL + 1;
        if (rp->NextVL >= nverts) rp->NextVL = 0;
        rp->NextVR = rp->CurVR - 1;
        if ((s32)rp->NextVR < 0) rp->NextVR = nverts - 1;
    }
    else
    {
        rp->NextVL = rp->CurVL - 1;
        if ((s32)rp->NextVL < 0) rp->NextVL = nverts - 1;
        rp->NextVR = rp->CurVR + 1;
        if (rp->NextVR >= nverts) rp->NextVR = 0;
    }

    if (ybot == ytop)
    {
        vtop = 0; vbot = 0;
        int i;

        i = 1;
        if (polygon->Vertices[i]->FinalPosition[0] < polygon->Vertices[vtop]->FinalPosition[0]) vtop = i;
        if (polygon->Vertices[i]->FinalPosition[0] > polygon->Vertices[vbot]->FinalPosition[0]) vbot = i;

        i = nverts - 1;
        if (polygon->Vertices[i]->FinalPosition[0] < polygon->Vertices[vtop]->FinalPosition[0]) vtop = i;
        if (polygon->Vertices[i]->FinalPosition[0] > polygon->Vertices[vbot]->FinalPosition[0]) vbot = i;

        rp->CurVL = vtop; rp->NextVL = vtop;
        rp->CurVR = vbot; rp->NextVR = vbot;

        rp->XL = rp->SlopeL.SetupDummy(polygon->Vertices[rp->CurVL]->FinalPosition[0], polygon->WBuffer);
        rp->XR = rp->SlopeR.SetupDummy(polygon->Vertices[rp->CurVR]->FinalPosition[0], polygon->WBuffer);
    }
    else
    {
        SetupPolygonLeftEdge(rp, ytop);
        SetupPolygonRightEdge(rp, ytop);
    }
}

void SoftRenderer3D::RenderShadowMaskScanline(RasterContext& ctx, RendererPolygon* rp, s32 y)
{
    Polygon* polygon = rp->PolyData;

    u32 polyattr = (polygon->Attr & 0x3F008000);
    if (!polygon->FacingView) polyattr |= (1<<4);

    u32 polyalpha = (polygon->Attr >> 16) & 0x1F;
    bool wireframe = (polyalpha == 0);

    bool (*fnDepthTest)(s32 dstz, s32 z, u32 dstattr);
    if (polygon->Attr & (1<<14))
        fnDepthTest = polygon->WBuffer ? DepthTest_Equal_W : DepthTest_Equal_Z;
    else if (polygon->FacingView)
        fnDepthTest = DepthTest_LessThan_FrontFacing;
    else
        fnDepthTest = DepthTest_LessThan;

    if (!ctx.PrevIsShadowMask)
        memset(&ctx.StencilBuffer[256 * (y&0x1)], 0, 256);

    ctx.PrevIsShadowMask = true;

    if (polygon->YTop != polygon->YBottom)
    {
        if (y >= polygon->Vertices[rp->NextVL]->FinalPosition[1] && rp->CurVL != polygon->VBottom)
        {
            SetupPolygonLeftEdge(rp, y);
        }

        if (y >= polygon->Vertices[rp->NextVR]->FinalPosition[1] && rp->CurVR != polygon->VBottom)
        {
            SetupPolygonRightEdge(rp, y);
        }
    }

    Vertex *vlcur, *vlnext, *vrcur, *vrnext;
    s32 xstart, xend;
    bool l_filledge, r_filledge;
    s32 l_edgelen, r_edgelen;
    s32 l_edgecov, r_edgecov;
    Interpolator<1>* interp_start;
    Interpolator<1>* interp_end;

    xstart = rp->XL;
    xend = rp->XR;

    s32 wl = rp->SlopeL.Interp.Interpolate(polygon->FinalW[rp->CurVL], polygon->FinalW[rp->NextVL]);
    s32 wr = rp->SlopeR.Interp.Interpolate(polygon->FinalW[rp->CurVR], polygon->FinalW[rp->NextVR]);

    s32 zl = rp->SlopeL.Interp.InterpolateZ(polygon->FinalZ[rp->CurVL], polygon->FinalZ[rp->NextVL]);
    s32 zr = rp->SlopeR.Interp.InterpolateZ(polygon->FinalZ[rp->CurVR], polygon->FinalZ[rp->NextVR]);

    // right vertical edges are pushed 1px to the left as long as either:
    // the left edge slope is not 0, or the span is not 0 pixels wide, and it is not at the leftmost pixel of the screen
    if (rp->SlopeR.Increment==0 && (rp->SlopeL.Increment!=0 || xstart != xend) && (xend != 0))
        xend--;

    // if the left and right edges are swapped, render backwards.
    if (xstart > xend)
    {
        vlcur = polygon->Vertices[rp->CurVR];
        vlnext = polygon->Vertices[rp->NextVR];
        vrcur = polygon->Vertices[rp->CurVL];
        vrnext = polygon->Vertices[rp->NextVL];

        interp_start = &rp->SlopeR.Interp;
        interp_end = &rp->SlopeL.Interp;

        rp->SlopeR.EdgeParams<true>(&l_edgelen, &l_edgecov);
        rp->SlopeL.EdgeParams<true>(&r_edgelen, &r_edgecov);

        std::swap(xstart, xend);
        std::swap(wl, wr);
        std::swap(zl, zr);

        // CHECKME: edge fill rules for swapped opaque shadow mask polygons
        if ((GPU3D.RenderDispCnt & ((1<<4)|(1<<5))) || ((polyalpha < 31) && (GPU3D.RenderDispCnt & (1<<3))) || wireframe)
        {
            l_filledge = true;
            r_filledge = true;
        }
        else
        {
            l_filledge = (rp->SlopeR.Negative || !rp->SlopeR.XMajor)
                || (y == polygon->YBottom-1) && rp->SlopeR.XMajor && (vlnext->FinalPosition[0] != vrnext->FinalPosition[0]);
            r_filledge = (!rp->SlopeL.Negative && rp->SlopeL.XMajor)
                || (!(rp->SlopeL.Negative && rp->SlopeL.XMajor) && rp->SlopeR.Increment==0)
                || (y == polygon->YBottom-1) && rp->SlopeL.XMajor && (vlnext->FinalPosition[0] != vrnext->FinalPosition[0]);
        }
    }
    else
    {
        vlcur = polygon->Vertices[rp->CurVL];
        vlnext = polygon->Vertices[rp->NextVL];
        vrcur = polygon->Vertices[rp->CurVR];
        vrnext = polygon->Vertices[rp->NextVR];

        interp_start = &rp->SlopeL.Interp;
        interp_end = &rp->SlopeR.Interp;

        rp->SlopeL.EdgeParams<false>(&l_edgelen, &l_edgecov);
        rp->SlopeR.EdgeParams<false>(&r_edgelen, &r_edgecov);

        // CHECKME: edge fill rules for unswapped opaque shadow mask polygons
        if ((GPU3D.RenderDispCnt & ((1<<4)|(1<<5))) || ((polyalpha < 31) && (GPU3D.RenderDispCnt & (1<<3))) || wireframe)
        {
            l_filledge = true;
            r_filledge = true;
        }
        else
        {
            l_filledge = ((rp->SlopeL.Negative || !rp->SlopeL.XMajor)
                || (y == polygon->YBottom-1) && rp->SlopeL.XMajor && (vlnext->FinalPosition[0] != vrnext->FinalPosition[0]))
                || (rp->SlopeL.Increment == rp->SlopeR.Increment) && (xstart+l_edgelen == xend+1);
            r_filledge = (!rp->SlopeR.Negative && rp->SlopeR.XMajor) || (rp->SlopeR.Increment==0)
                || (y == polygon->YBottom-1) && rp->SlopeR.XMajor && (vlnext->FinalPosition[0] != vrnext->FinalPosition[0]);
        }
    }

    // color/texcoord attributes aren't needed for shadow masks
    // all the pixels are guaranteed to have the same alpha
    // even if a texture is used (decal blending is used for shadows)
    // similarly, we can perform alpha test early (checkme)

    if (wireframe) polyalpha = 31;
    if (polyalpha <= GPU3D.RenderAlphaRef) return;

    // in wireframe mode, there are special rules for equal Z (TODO)

    int yedge = 0;
    if (y == polygon->YTop)           yedge = 0x4;
    else if (y == polygon->YBottom-1) yedge = 0x8;
    int edge;

    s32 x = xstart;
    Interpolator<0> interpX(xstart, xend+1, wl, wr, polygon->WBuffer);

    if (x < 0) x = 0;
    s32 xlimit;

    // for shadow masks: set stencil bits where the depth test fails.
    // draw nothing.

    // part 1: left edge
    edge = yedge | 0x1;
    xlimit = xstart+l_edgelen;
    if (xlimit > xend+1) xlimit = xend+1;
    if (xlimit > 256) xlimit = 256;

    if (!l_filledge) x = xlimit;
    else
    for (; x < xlimit; x++)
    {
        u32 pixeladdr = FirstPixelOffset + (y*ScanlineWidth) + x;

        interpX.SetX(x);

        s32 z = interpX.InterpolateZ(zl, zr);
        u32 dstattr = AttrBuffer[pixeladdr];

        if (!fnDepthTest(DepthBuffer[pixeladdr], z, dstattr))
            ctx.StencilBuffer[256*(y&0x1) + x] = 1;

        if (dstattr & 0xF)
        {
            pixeladdr += BufferSize;
            if (!fnDepthTest(DepthBuffer[pixeladdr], z, AttrBuffer[pixeladdr]))
                ctx.StencilBuffer[256*(y&0x1) + x] |= 0x2;
        }
    }

    // part 2: polygon inside
    edge = yedge;
    xlimit = xend-r_edgelen+1;
    if (xlimit > xend+1) xlimit = xend+1;
    if (xlimit > 256) xlimit = 256;
    if (wireframe && !edge) x = std::max(x, xlimit);
    else for (; x < xlimit; x++)
    {
        u32 pixeladdr = FirstPixelOffset + (y*ScanlineWidth) + x;

        interpX.SetX(x);

        s32 z = interpX.InterpolateZ(zl, zr);
        u32 dstattr = AttrBuffer[pixeladdr];

        if (!fnDepthTest(DepthBuffer[pixeladdr], z, dstattr))
            ctx.StencilBuffer[256*(y&0x1) + x] = 1;

        if (dstattr & 0xF)
        {
            pixeladdr += BufferSize;
            if (!fnDepthTest(DepthBuffer[pixeladdr], z, AttrBuffer[pixeladdr]))
                ctx.StencilBuffer[256*(y&0x1) + x] |= 0x2;
        }
    }

    // part 3: right edge
    edge = yedge | 0x2;
    xlimit = xend+1;
    if (xlimit > 256) xlimit = 256;

    if (r_filledge)
    for (; x < xlimit; x++)
    {
        u32 pixeladdr = FirstPixelOffset + (y*ScanlineWidth) + x;

        interpX.SetX(x);

        s32 z = interpX.InterpolateZ(zl, zr);
        u32 dstattr = AttrBuffer[pixeladdr];

        if (!fnDepthTest(DepthBuffer[pixeladdr], z, dstattr))
            ctx.StencilBuffer[256*(y&0x1) + x] = 1;

        if (dstattr & 0xF)
        {
            pixeladdr += BufferSize;
            if (!fnDepthTest(DepthBuffer[pixeladdr], z, AttrBuffer[pixeladdr]))
                ctx.StencilBuffer[256*(y&0x1) + x] |= 0x2;
        }
    }

    rp->XL = rp->SlopeL.Step();
    rp->XR = rp->SlopeR.Step();
}

void SoftRenderer3D::RenderPolygonScanline(RasterContext& ctx, RendererPolygon* rp, s32 y)
{
    Polygon* polygon = rp->PolyData;

    u32 polyattr = (polygon->Attr & 0x3F008000);
    if (!polygon->FacingView) polyattr |= (1<<4);

    u32 polyalpha = (polygon->Attr >> 16) & 0x1F;
    bool wireframe = (polyalpha == 0);

    ctx.PrevIsShadowMask = false;

    if (polygon->YTop != polygon->YBottom)
    {
        if (y >= polygon->Vertices[rp->NextVL]->FinalPosition[1] && rp->CurVL != polygon->VBottom)
        {
            SetupPolygonLeftEdge(rp, y);
        }

        if (y >= polygon->Vertices[rp->NextVR]->FinalPosition[1] && rp->CurVR != polygon->VBottom)
        {
            SetupPolygonRightEdge(rp, y);
        }
    }

    Vertex *vlcur, *vlnext, *vrcur, *vrnext;
    s32 xstart, xend;
    bool l_filledge, r_filledge;
    s32 l_edgelen, r_edgelen;
    s32 l_edgecov, r_edgecov;
    Interpolator<1>* interp_start;
    Interpolator<1>* interp_end;

    xstart = rp->XL;
    xend = rp->XR;

    s32 wl = rp->SlopeL.Interp.Interpolate(polygon->FinalW[rp->CurVL], polygon->FinalW[rp->NextVL]);
    s32 wr = rp->SlopeR.Interp.Interpolate(polygon->FinalW[rp->CurVR], polygon->FinalW[rp->NextVR]);

    s32 zl = rp->SlopeL.Interp.InterpolateZ(polygon->FinalZ[rp->CurVL], polygon->FinalZ[rp->NextVL]);
    s32 zr = rp->SlopeR.Interp.InterpolateZ(polygon->FinalZ[rp->CurVR], polygon->FinalZ[rp->NextVR]);

    // right vertical edges are pushed 1px to the left as long as either:
    // the left edge slope is not 0, or the span is not 0 pixels wide, and it is not at the leftmost pixel of the screen
    if (rp->SlopeR.Increment==0 && (rp->SlopeL.Increment!=0 || xstart != xend) && (xend != 0))
        xend--;

    // if the left and right edges are swapped, render backwards.
    // on hardware, swapped edges seem to break edge length calculation,
    // causing X-major edges to be rendered wrong when filled,
    // and resulting in buggy looking anti-aliasing on X-major edges

    if (xstart > xend)
    {
        vlcur = polygon->Vertices[rp->CurVR];
        vlnext = polygon->Vertices[rp->NextVR];
        vrcur = polygon->Vertices[rp->CurVL];
        vrnext = polygon->Vertices[rp->NextVL];

        interp_start = &rp->SlopeR.Interp;
        interp_end = &rp->SlopeL.Interp;

        rp->SlopeR.EdgeParams<true>(&l_edgelen, &l_edgecov);
        rp->SlopeL.EdgeParams<true>(&r_edgelen, &r_edgecov);

        std::swap(xstart, xend);
        std::swap(wl, wr);
        std::swap(zl, zr);

        // edge fill rules for swapped opaque edges:
        // * right edge is filled if slope > 1, or if the left edge = 0, but is never filled if it is < -1
        // * left edge is filled if slope <= 1
        // * the bottom-most pixel of negative x-major slopes are filled if they are next to a flat bottom edge
        // edges are always filled if antialiasing/edgemarking are enabled,
        // if the pixels are translucent and alpha blending is enabled, or if the polygon is wireframe
        // checkme: do swapped line polygons exist?
        if ((GPU3D.RenderDispCnt & ((1<<4)|(1<<5))) || ((polyalpha < 31) && (GPU3D.RenderDispCnt & (1<<3))) || wireframe)
        {
            l_filledge = true;
            r_filledge = true;
        }
        else
        {
            l_filledge = (rp->SlopeR.Negative || !rp->SlopeR.XMajor)
                || (y == polygon->YBottom-1) && rp->SlopeR.XMajor && (vlnext->FinalPosition[0] != vrnext->FinalPosition[0]);
            r_filledge = (!rp->SlopeL.Negative && rp->SlopeL.XMajor)
                || (!(rp->SlopeL.Negative && rp->SlopeL.XMajor) && rp->SlopeR.Increment==0)
                || (y == polygon->YBottom-1) && rp->SlopeL.XMajor && (vlnext->FinalPosition[0] != vrnext->FinalPosition[0]);
        }
    }
    else
    {
        vlcur = polygon->Vertices[rp->CurVL];
        vlnext = polygon->Vertices[rp->NextVL];
        vrcur = polygon->Vertices[rp->CurVR];
        vrnext = polygon->Vertices[rp->NextVR];

        interp_start = &rp->SlopeL.Interp;
        interp_end = &rp->SlopeR.Interp;

        rp->SlopeL.EdgeParams<false>(&l_edgelen, &l_edgecov);
        rp->SlopeR.EdgeParams<false>(&r_edgelen, &r_edgecov);

        // edge fill rules for unswapped opaque edges:
        // * right edge is filled if slope > 1
        // * left edge is filled if slope <= 1
        // * edges with slope = 0 are always filled
        // * the bottom-most pixel of negative x-major slopes are filled if they are next to a flat bottom edge
        // * edges are filled if both sides are identical and fully overlapping
        // edges are always filled if antialiasing/edgemarking are enabled,
        // if the pixels are translucent and alpha blending is enabled, or if the polygon is wireframe
        if ((GPU3D.RenderDispCnt & ((1<<4)|(1<<5))) || ((polyalpha < 31) && (GPU3D.RenderDispCnt & (1<<3))) || wireframe)
        {
            l_filledge = true;
            r_filledge = true;
        }
        else
        {
            l_filledge = ((rp->SlopeL.Negative || !rp->SlopeL.XMajor)
                || (y == polygon->YBottom-1) && rp->SlopeL.XMajor && (vlnext->FinalPosition[0] != vrnext->FinalPosition[0]))
                || (rp->SlopeL.Increment == rp->SlopeR.Increment) && (xstart+l_edgelen == xend+1);
            r_filledge = (!rp->SlopeR.Negative && rp->SlopeR.XMajor) || (rp->SlopeR.Increment==0)
                || (y == polygon->YBottom-1) && rp->SlopeR.XMajor && (vlnext->FinalPosition[0] != vrnext->FinalPosition[0]);
        }
    }

    // interpolate attributes along Y

    s32 rl = interp_start->Interpolate(vlcur->FinalColor[0], vlnext->FinalColor[0]);
    s32 gl = interp_start->Interpolate(vlcur->FinalColor[1], vlnext->FinalColor[1]);
    s32 bl = interp_start->Interpolate(vlcur->FinalColor[2], vlnext->FinalColor[2]);

    s32 sl = interp_start->Interpolate(vlcur->TexCoords[0], vlnext->TexCoords[0]);
    s32 tl = interp_start->Interpolate(vlcur->TexCoords[1], vlnext->TexCoords[1]);

    s32 rr = interp_end->Interpolate(vrcur->FinalColor[0], vrnext->FinalColor[0]);
    s32 gr = interp_end->Interpolate(vrcur->FinalColor[1], vrnext->FinalColor[1]);
    s32 br = interp_end->Interpolate(vrcur->FinalColor[2], vrnext->FinalColor[2]);

    s32 sr = interp_end->Interpolate(vrcur->TexCoords[0], vrnext->TexCoords[0]);
    s32 tr = interp_end->Interpolate(vrcur->TexCoords[1], vrnext->TexCoords[1]);

    // in wireframe mode, there are special rules for equal Z (TODO)

    SpanSetup span;
    span.XStart = xstart;
    span.XEnd = xend;
    span.LFillEdge = l_filledge;
    span.RFillEdge = r_filledge;
    span.LEdgeLen = l_edgelen;
    span.REdgeLen = r_edgelen;
    span.LEdgeCov = l_edgecov;
    span.REdgeCov = r_edgecov;

    span.YEdge = 0;
    if (y == polygon->YTop)           span.YEdge = 0x4;
    else if (y == polygon->YBottom-1) span.YEdge = 0x8;

    span.PolyAttr = polyattr;
    span.Wireframe = wireframe;

    span.InterpX.Setup(xstart, xend+1, wl, wr, polygon->WBuffer);
    span.ZL = zl; span.ZR = zr;
    span.RL = rl; span.GL = gl; span.BL = bl; span.SL = sl; span.TL = tl;
    span.RR = rr; span.GR = gr; span.BR = br; span.SR = sr; span.TR = tr;

    // the span is never empty, since xstart <= xend
    if (!span.InterpX.IsLinear())
    {
        DrawSpan<SpanMode::Perspective>(ctx, polygon, span, y);
    }
    else
    {
        SpanInterpolator interp(span.InterpX, zl, zr);
        if (interp.LinearExact(rl, rr) && interp.LinearExact(gl, gr) && interp.LinearExact(bl, br)
            && interp.LinearExact(sl, sr) && interp.LinearExact(tl, tr))
            DrawSpan<SpanMode::Linear>(ctx, polygon, span, y);
        else
            DrawSpan<SpanMode::Generic>(ctx, polygon, span, y);
    }

    rp->XL = rp->SlopeL.Step();
    rp->XR = rp->SlopeR.Step();
}

enum class DepthTestMode { Equal_Z, Equal_W, LessThan, LessThan_FrontFacing };

[[gnu::always_inline]] static inline bool DepthTest(DepthTestMode mode, s32 dstz, s32 z, u32 dstattr)
{
    switch (mode)
    {
    case DepthTestMode::Equal_Z: return DepthTest_Equal_Z(dstz, z, dstattr);
    case DepthTestMode::Equal_W: return DepthTest_Equal_W(dstz, z, dstattr);
    case DepthTestMode::LessThan: return DepthTest_LessThan(dstz, z, dstattr);
    default: return DepthTest_LessThan_FrontFacing(dstz, z, dstattr);
    }
}

template <SoftRenderer3D::SpanMode Mode>
void SoftRenderer3D::DrawSpan(RasterContext& ctx, const Polygon* polygon, const SpanSetup& span, s32 y)
{
    // the pixels are written through u32 pointers, which may alias the polygon and the render state,
    // so everything that is needed for them is read up front
    const u32 polyattr = span.PolyAttr;
    const bool isshadow = polygon->IsShadow;
    const bool translucentdepth = polygon->Attr & (1<<11);
    const bool antialias = GPU3D.RenderDispCnt & (1<<4);
    const u32 alpharef = GPU3D.RenderAlphaRef;
    const u8* stencil = &ctx.StencilBuffer[256*(y&0x1)];
    const u32 lineaddr = FirstPixelOffset + (y*ScanlineWidth);

    PixelShading shading;
    shading.BlendMode = (polygon->Attr >> 4) & 0x3;
    shading.PolyAlpha = (polygon->Attr >> 16) & 0x1F;
    shading.Wireframe = (shading.PolyAlpha == 0);
    shading.Highlight = GPU3D.RenderDispCnt & (1<<1);
    shading.Textured = (GPU3D.RenderDispCnt & (1<<0)) && (((polygon->TexParam >> 26) & 0x7) != 0);
    shading.TexParam = polygon->TexParam;
    shading.TexPalette = polygon->TexPalette;

    DepthTestMode depthtest;
    if (polygon->Attr & (1<<14))
        depthtest = polygon->WBuffer ? DepthTestMode::Equal_W : DepthTestMode::Equal_Z;
    else if (polygon->FacingView)
        depthtest = DepthTestMode::LessThan_FrontFacing;
    else
        depthtest = DepthTestMode::LessThan;

    Interpolator<0> interpX = span.InterpX;
    const SpanInterpolator spanX(span.InterpX, span.ZL, span.ZR);
    const SpanInterpolator::Attr ar(span.RL, span.RR), ag(span.GL, span.GR), ab(span.BL, span.BR);
    const SpanInterpolator::Attr as(span.SL, span.SR), at(span.TL, span.TR);
    const SpanInterpolator::LinearAttr lr = spanX.Linear(span.RL, span.RR), lg = spanX.Linear(span.GL, span.GR);
    const SpanInterpolator::LinearAttr lb = spanX.Linear(span.BL, span.BR);
    const SpanInterpolator::LinearAttr ls = spanX.Linear(span.SL, span.SR), lt = spanX.Linear(span.TL, span.TR);
    const bool wbuffer = polygon->WBuffer;

    s32 xcov = 0;

    // one pixel of the left edge (part 0), the inside (part 1) or the right edge (part 2)
    auto pixel = [&](s32 x, auto part, int edge, s32 edgecov) __attribute__((always_inline))
    {
        constexpr int Part = decltype(part)::value;

        u32 pixeladdr = lineaddr + x;
        u32 dstattr = AttrBuffer[pixeladdr];

        // check stencil buffer for shadows
        if (isshadow)
        {
            u8 st = stencil[x];
            if (!st)
                return;
            if (!(st & 0x1))
                pixeladdr += BufferSize;
            if (!(st & 0x2))
                dstattr &= ~0xF; // quick way to prevent drawing the shadow under antialiased edges
        }

        u32 factor = 0;
        s32 z;
        if constexpr (Mode == SpanMode::Perspective)
        {
            factor = spanX.Factor(x);
            z = spanX.Z(x, factor);
        }
        else if constexpr (Mode == SpanMode::Linear)
        {
            // W-buffering still uses the perspective factor
            if (wbuffer) factor = spanX.Factor(x);
            z = spanX.Z(x, factor);
        }
        else
        {
            interpX.SetX(x);
            z = interpX.InterpolateZ(span.ZL, span.ZR);
        }

        // if depth test against the topmost pixel fails, test
        // against the pixel underneath
        if (!DepthTest(depthtest, DepthBuffer[pixeladdr], z, dstattr))
        {
            if (!(dstattr & 0xF) || pixeladdr >= BufferSize) return;

            pixeladdr += BufferSize;
            dstattr = AttrBuffer[pixeladdr];
            if (!DepthTest(depthtest, DepthBuffer[pixeladdr], z, dstattr))
                return;
        }

        u32 vr, vg, vb;
        s16 s, t;
        if constexpr (Mode == SpanMode::Linear)
        {
            vr = spanX.At(lr, x);
            vg = spanX.At(lg, x);
            vb = spanX.At(lb, x);

            s = spanX.At(ls, x);
            t = spanX.At(lt, x);
        }
        else if constexpr (Mode == SpanMode::Perspective)
        {
            vr = ar.At(factor);
            vg = ag.At(factor);
            vb = ab.At(factor);

            s = as.At(factor);
            t = at.At(factor);
        }
        else
        {
            vr = interpX.Interpolate(span.RL, span.RR);
            vg = interpX.Interpolate(span.GL, span.GR);
            vb = interpX.Interpolate(span.BL, span.BR);

            s = interpX.Interpolate(span.SL, span.SR);
            t = interpX.Interpolate(span.TL, span.TR);
        }

        u32 color = RenderPixel(shading, vr>>3, vg>>3, vb>>3, s, t);
        u8 alpha = color >> 24;

        // alpha test
        if (alpha <= alpharef) return;

        if (alpha == 31)
        {
            u32 attr = polyattr | edge;

            if (antialias && ((Part != 1) || (attr & 0xF)))
            {
                // anti-aliasing: all edges are rendered

                if constexpr (Part == 1)
                {
                    // set coverage to avoid black lines from anti-aliasing
                    attr |= (0x1F << 8);
                }
                else
                {
                    // calculate coverage
                    s32 cov = edgecov;
                    if (cov & (1<<31))
                    {
                        if constexpr (Part == 0)
                        {
                            cov = xcov >> 5;
                            if (cov > 31) cov = 31;
                        }
                        else
                        {
                            cov = 0x1F - (xcov >> 5);
                            if (cov < 0) cov = 0;
                        }
                        xcov += (edgecov & 0x3FF);
                    }
                    attr |= (cov << 8);
                }

                // push old pixel down if needed
                if (pixeladdr < BufferSize)
                {
                    ColorBuffer[pixeladdr+BufferSize] = ColorBuffer[pixeladdr];
                    DepthBuffer[pixeladdr+BufferSize] = DepthBuffer[pixeladdr];
                    AttrBuffer[pixeladdr+BufferSize] = AttrBuffer[pixeladdr];
                }
            }

            DepthBuffer[pixeladdr] = z;
            ColorBuffer[pixeladdr] = color;
            AttrBuffer[pixeladdr] = attr;
        }
        else
        {
            if (!translucentdepth) z = -1;
            PlotTranslucentPixel(pixeladdr, color, z, polyattr, isshadow);

            // blend with bottom pixel too, if needed
            if ((dstattr & 0xF) && (pixeladdr < BufferSize))
                PlotTranslucentPixel(pixeladdr+BufferSize, color, z, polyattr, isshadow);
        }
    };

    s32 x = span.XStart;
    if (x < 0) x = 0;
    s32 xlimit;
    int edge;

    // part 1: left edge
    edge = span.YEdge | 0x1;
    xlimit = span.XStart+span.LEdgeLen;
    if (xlimit > span.XEnd+1) xlimit = span.XEnd+1;
    if (xlimit > 256) xlimit = 256;
    if (span.LEdgeCov & (1<<31))
    {
        xcov = (span.LEdgeCov >> 12) & 0x3FF;
        if (xcov == 0x3FF) xcov = 0;
    }

    if (!span.LFillEdge) x = xlimit;
    else
    for (; x < xlimit; x++)
        pixel(x, std::integral_constant<int, 0>(), edge, span.LEdgeCov);

    // part 2: polygon inside
    edge = span.YEdge;
    xlimit = span.XEnd-span.REdgeLen+1;
    if (xlimit > span.XEnd+1) xlimit = span.XEnd+1;
    if (xlimit > 256) xlimit = 256;

    if (span.Wireframe && !edge) x = std::max(x, xlimit);
    else
    for (; x < xlimit; x++)
        pixel(x, std::integral_constant<int, 1>(), edge, 0);

    // part 3: right edge
    edge = span.YEdge | 0x2;
    xlimit = span.XEnd+1;
    if (xlimit > 256) xlimit = 256;
    if (span.REdgeCov & (1<<31))
    {
        xcov = (span.REdgeCov >> 12) & 0x3FF;
        if (xcov == 0x3FF) xcov = 0;
    }

    if (span.RFillEdge)
    for (; x < xlimit; x++)
        pixel(x, std::integral_constant<int, 2>(), edge, span.REdgeCov);
}

void SoftRenderer3D::RenderScanline(RasterContext& ctx, s32 y, int npolys)
{
    for (int i = 0; i < npolys; i++)
    {
        const LineRange& lines = ctx.Ranges[i];
        if (y < lines.Top || y >= lines.Bottom)
            continue;

        RendererPolygon* rp = &ctx.Polygons[i];
        if (rp->PolyData->IsShadowMask)
            RenderShadowMaskScanline(ctx, rp, y);
        else
            RenderPolygonScanline(ctx, rp, y);
    }
}

u32 SoftRenderer3D::CalculateFogDensity(u32 pixeladdr) const
{
    u32 z = DepthBuffer[pixeladdr];
    u32 densityid, densityfrac;

    if (z < GPU3D.RenderFogOffset)
    {
        densityid = 0;
        densityfrac = 0;
    }
    else
    {
        // technically: Z difference is shifted right by two, then shifted left by fog shift
        // then bit 0-16 are the fractional part and bit 17-31 are the density index
        // on hardware, the final value can overflow the 32-bit range with a shift big enough,
        // causing fog to 'wrap around' and accidentally apply to larger Z ranges

        z -= GPU3D.RenderFogOffset;
        z = (z >> 2) << GPU3D.RenderFogShift;

        densityid = z >> 17;
        if (densityid >= 32)
        {
            densityid = 32;
            densityfrac = 0;
        }
        else
            densityfrac = z & 0x1FFFF;
    }

    // checkme (may be too precise?)
    u32 density =
        ((GPU3D.RenderFogDensityTable[densityid] * (0x20000-densityfrac)) +
         (GPU3D.RenderFogDensityTable[densityid+1] * densityfrac)) >> 17;
    if (density >= 127) density = 128;

    return density;
}

void SoftRenderer3D::ScanlineFinalPass(s32 y)
{
    // to consider:
    // clearing all polygon fog flags if the master flag isn't set?
    // merging all final pass loops into one?

    if (GPU3D.RenderDispCnt & (1<<5))
    {
        // edge marking
        // only applied to topmost pixels

        for (int x = 0; x < 256; x++)
        {
            u32 pixeladdr = FirstPixelOffset + (y*ScanlineWidth) + x;

            u32 attr = AttrBuffer[pixeladdr];
            if (!(attr & 0xF)) continue;

            u32 polyid = attr >> 24; // opaque polygon IDs are used for edgemarking
            u32 z = DepthBuffer[pixeladdr];

            if (((polyid != (AttrBuffer[pixeladdr-1] >> 24)) && (z < DepthBuffer[pixeladdr-1])) ||
                ((polyid != (AttrBuffer[pixeladdr+1] >> 24)) && (z < DepthBuffer[pixeladdr+1])) ||
                ((polyid != (AttrBuffer[pixeladdr-ScanlineWidth] >> 24)) && (z < DepthBuffer[pixeladdr-ScanlineWidth])) ||
                ((polyid != (AttrBuffer[pixeladdr+ScanlineWidth] >> 24)) && (z < DepthBuffer[pixeladdr+ScanlineWidth])))
            {
                u16 edgecolor = GPU3D.RenderEdgeTable[polyid >> 3];
                u32 edgeR = (edgecolor << 1) & 0x3E; if (edgeR) edgeR++;
                u32 edgeG = (edgecolor >> 4) & 0x3E; if (edgeG) edgeG++;
                u32 edgeB = (edgecolor >> 9) & 0x3E; if (edgeB) edgeB++;

                ColorBuffer[pixeladdr] = edgeR | (edgeG << 8) | (edgeB << 16) | (ColorBuffer[pixeladdr] & 0xFF000000);

                // break antialiasing coverage (checkme)
                AttrBuffer[pixeladdr] = (AttrBuffer[pixeladdr] & 0xFFFFE0FF) | 0x00001000;
            }
        }
    }

    if (GPU3D.RenderDispCnt & (1<<7))
    {
        // fog

        // hardware testing shows that the fog step is 0x80000>>SHIFT
        // basically, the depth values used in GBAtek need to be
        // multiplied by 0x200 to match Z-buffer values

        // fog is applied to the topmost two pixels, which is required for
        // proper antialiasing

        // TODO: check the 'fog alpha glitch with small Z' GBAtek talks about

        bool fogcolor = !(GPU3D.RenderDispCnt & (1<<6));

        u32 fogR = (GPU3D.RenderFogColor << 1) & 0x3E; if (fogR) fogR++;
        u32 fogG = (GPU3D.RenderFogColor >> 4) & 0x3E; if (fogG) fogG++;
        u32 fogB = (GPU3D.RenderFogColor >> 9) & 0x3E; if (fogB) fogB++;
        u32 fogA = (GPU3D.RenderFogColor >> 16) & 0x1F;

        for (int x = 0; x < 256; x++)
        {
            u32 pixeladdr = FirstPixelOffset + (y*ScanlineWidth) + x;
            u32 density, srccolor, srcR, srcG, srcB, srcA;

            u32 attr = AttrBuffer[pixeladdr];
            if (attr & (1<<15))
            {
                density = CalculateFogDensity(pixeladdr);

                srccolor = ColorBuffer[pixeladdr];
                srcR = srccolor & 0x3F;
                srcG = (srccolor >> 8) & 0x3F;
                srcB = (srccolor >> 16) & 0x3F;
                srcA = (srccolor >> 24) & 0x1F;

                if (fogcolor)
                {
                    srcR = ((fogR * density) + (srcR * (128-density))) >> 7;
                    srcG = ((fogG * density) + (srcG * (128-density))) >> 7;
                    srcB = ((fogB * density) + (srcB * (128-density))) >> 7;
                }

                srcA = ((fogA * density) + (srcA * (128-density))) >> 7;

                ColorBuffer[pixeladdr] = srcR | (srcG << 8) | (srcB << 16) | (srcA << 24);
            }

            // fog for lower pixel
            // TODO: make this code nicer, but avoid using a loop

            if (!(attr & 0xF)) continue;
            pixeladdr += BufferSize;

            attr = AttrBuffer[pixeladdr];
            if (!(attr & (1<<15))) continue;

            density = CalculateFogDensity(pixeladdr);

            srccolor = ColorBuffer[pixeladdr];
            srcR = srccolor & 0x3F;
            srcG = (srccolor >> 8) & 0x3F;
            srcB = (srccolor >> 16) & 0x3F;
            srcA = (srccolor >> 24) & 0x1F;

            if (fogcolor)
            {
                srcR = ((fogR * density) + (srcR * (128-density))) >> 7;
                srcG = ((fogG * density) + (srcG * (128-density))) >> 7;
                srcB = ((fogB * density) + (srcB * (128-density))) >> 7;
            }

            srcA = ((fogA * density) + (srcA * (128-density))) >> 7;

            ColorBuffer[pixeladdr] = srcR | (srcG << 8) | (srcB << 16) | (srcA << 24);
        }
    }

    if (GPU3D.RenderDispCnt & (1<<4))
    {
        // anti-aliasing

        // edges were flagged and their coverages calculated during rendering
        // this is where such edge pixels are blended with the pixels underneath

        for (int x = 0; x < 256; x++)
        {
            u32 pixeladdr = FirstPixelOffset + (y*ScanlineWidth) + x;

            u32 attr = AttrBuffer[pixeladdr];
            if (!(attr & 0xF)) continue;

            u32 coverage = (attr >> 8) & 0x1F;
            if (coverage == 0x1F) continue;

            if (coverage == 0)
            {
                ColorBuffer[pixeladdr] = ColorBuffer[pixeladdr+BufferSize];
                continue;
            }

            u32 topcolor = ColorBuffer[pixeladdr];
            u32 topR = topcolor & 0x3F;
            u32 topG = (topcolor >> 8) & 0x3F;
            u32 topB = (topcolor >> 16) & 0x3F;
            u32 topA = (topcolor >> 24) & 0x1F;

            u32 botcolor = ColorBuffer[pixeladdr+BufferSize];
            u32 botR = botcolor & 0x3F;
            u32 botG = (botcolor >> 8) & 0x3F;
            u32 botB = (botcolor >> 16) & 0x3F;
            u32 botA = (botcolor >> 24) & 0x1F;

            coverage++;

            // only blend color if the bottom pixel isn't fully transparent
            if (botA > 0)
            {
                topR = ((topR * coverage) + (botR * (32-coverage))) >> 5;
                topG = ((topG * coverage) + (botG * (32-coverage))) >> 5;
                topB = ((topB * coverage) + (botB * (32-coverage))) >> 5;
            }

            // alpha is always blended
            topA = ((topA * coverage) + (botA * (32-coverage))) >> 5;

            ColorBuffer[pixeladdr] = topR | (topG << 8) | (topB << 16) | (topA << 24);
        }
    }
}

void SoftRenderer3D::ClearBuffers()
{
    u32 clearz = ((GPU3D.RenderClearAttr2 & 0x7FFF) * 0x200) + 0x1FF;
    u32 polyid = GPU3D.RenderClearAttr1 & 0x3F000000; // this sets the opaque polygonID

    // fill screen borders for edge marking

    for (int x = 0; x < ScanlineWidth; x++)
    {
        ColorBuffer[x] = 0;
        DepthBuffer[x] = clearz;
        AttrBuffer[x] = polyid;
    }

    for (int x = ScanlineWidth; x < ScanlineWidth*193; x+=ScanlineWidth)
    {
        ColorBuffer[x] = 0;
        DepthBuffer[x] = clearz;
        AttrBuffer[x] = polyid;
        ColorBuffer[x+257] = 0;
        DepthBuffer[x+257] = clearz;
        AttrBuffer[x+257] = polyid;
    }

    for (int x = ScanlineWidth*193; x < ScanlineWidth*194; x++)
    {
        ColorBuffer[x] = 0;
        DepthBuffer[x] = clearz;
        AttrBuffer[x] = polyid;
    }

    // clear the screen

    if (GPU3D.RenderDispCnt & (1<<14))
    {
        u8 xoff = (GPU3D.RenderClearAttr2 >> 16) & 0xFF;
        u8 yoff = (GPU3D.RenderClearAttr2 >> 24) & 0xFF;

        for (int y = 0; y < ScanlineWidth*192; y+=ScanlineWidth)
        {
            for (int x = 0; x < 256; x++)
            {
                u16 val2 = GPU.ReadVRAMFlat_Texture<u16>(0x40000 + (yoff << 9) + (xoff << 1));
                u16 val3 = GPU.ReadVRAMFlat_Texture<u16>(0x60000 + (yoff << 9) + (xoff << 1));

                // TODO: confirm color conversion
                u32 r = (val2 << 1) & 0x3E; if (r) r++;
                u32 g = (val2 >> 4) & 0x3E; if (g) g++;
                u32 b = (val2 >> 9) & 0x3E; if (b) b++;
                u32 a = (val2 & 0x8000) ? 0x1F000000 : 0;
                u32 color = r | (g << 8) | (b << 16) | a;

                u32 z = ((val3 & 0x7FFF) * 0x200) + 0x1FF;

                u32 pixeladdr = FirstPixelOffset + y + x;
                ColorBuffer[pixeladdr] = color;
                DepthBuffer[pixeladdr] = z;
                AttrBuffer[pixeladdr] = polyid | (val3 & 0x8000);

                xoff++;
            }

            yoff++;
        }
    }
    else
    {
        // TODO: confirm color conversion
        u32 r = (GPU3D.RenderClearAttr1 << 1) & 0x3E; if (r) r++;
        u32 g = (GPU3D.RenderClearAttr1 >> 4) & 0x3E; if (g) g++;
        u32 b = (GPU3D.RenderClearAttr1 >> 9) & 0x3E; if (b) b++;
        u32 a = (GPU3D.RenderClearAttr1 >> 16) & 0x1F;
        u32 color = r | (g << 8) | (b << 16) | (a << 24);

        polyid |= (GPU3D.RenderClearAttr1 & 0x8000);

        for (int y = 0; y < ScanlineWidth*192; y+=ScanlineWidth)
        {
            for (int x = 0; x < 256; x++)
            {
                u32 pixeladdr = FirstPixelOffset + y + x;
                ColorBuffer[pixeladdr] = color;
                DepthBuffer[pixeladdr] = clearz;
                AttrBuffer[pixeladdr] = polyid;
            }
        }
    }
}

void SoftRenderer3D::RenderPolygons(bool threaded, Polygon** polygons, int npolys)
{
    if (threaded && SplitThreadRunning.load(std::memory_order_relaxed))
    {
        RenderPolygonsSplit(polygons, npolys);
        return;
    }

    int j = 0;
    for (int i = 0; i < npolys; i++)
    {
        if (polygons[i]->Degenerate) continue;
        PolygonRanges[j] = PolygonLines(polygons[i]);
        SetupPolygon(&PolygonList[j++], polygons[i]);
    }

    RasterContext ctx {PolygonList, PolygonRanges, StencilBuffer, PrevIsShadowMask};

    RenderScanline(ctx, 0, j);

    for (s32 y = 1; y < 192; y++)
    {
        if (threaded && CancelFrame.load(std::memory_order_relaxed))
            break;

        RenderScanline(ctx, y, j);
        ScanlineFinalPass(y-1);

        if (threaded)
            // Notify the main thread that we're done with a scanline.
            Platform::Semaphore_Post(Sema_ScanlineCount);
    }

    ScanlineFinalPass(191);

    PrevIsShadowMask = ctx.PrevIsShadowMask;

    if (threaded)
        // If this renderer is threaded, notify the main thread that we're done with the frame.
        Platform::Semaphore_Post(Sema_ScanlineCount);
}

s32 SoftRenderer3D::ChooseSplitLine(Polygon** polygons, int npolys) const
{
    // estimate the work per scanline: a fixed cost per polygon plus its width
    s32 delta[193] {};
    s64 total = 0;
    for (int i = 0; i < npolys; i++)
    {
        const Polygon* polygon = polygons[i];
        if (polygon->Degenerate) continue;

        s32 ytop = std::clamp<s32>(polygon->YTop, 0, 191);
        s32 ybot = (polygon->YBottom == polygon->YTop) ? polygon->YTop + 1 : polygon->YBottom;
        ybot = std::clamp<s32>(ybot, ytop + 1, 192);

        s32 xmin = 256, xmax = 0;
        for (u32 v = 0; v < polygon->NumVertices; v++)
        {
            s32 x = polygon->Vertices[v]->FinalPosition[0];
            xmin = std::min(xmin, x);
            xmax = std::max(xmax, x);
        }
        s32 cost = 32 + std::clamp<s32>(xmax - xmin, 0, 256);

        delta[ytop] += cost;
        delta[ybot] -= cost;
        total += (s64)cost * (ybot - ytop);
    }

    // the top half is needed first, so it gets a little less than half
    s64 target = total * 9 / 20;
    s64 sum = 0;
    s32 cost = 0;
    s32 line = 0;
    for (; line < 192; line++)
    {
        cost += delta[line];
        if (sum + cost > target) break;
        sum += cost;
    }

    return std::clamp<s32>(line, 8, 184);
}

void SoftRenderer3D::RenderPolygonsSplit(Polygon** polygons, int npolys)
{
    SplitLine = ChooseSplitLine(polygons, npolys);
    TopHalfDone = false;
    TopReady = 0;
    BoundaryReady = false;
    BottomReady = 0;
    LinesPublished = 0;
    Platform::Semaphore_Reset(Sema_TopHalfDone);

    // the split thread sets up its own polygons from the same list
    SplitPolygons = polygons;
    SplitNumPolygons = npolys;
    SplitStartPrevIsShadowMask = PrevIsShadowMask;
    Platform::Semaphore_Post(Sema_SplitStart);

    int j = 0;
    for (int i = 0; i < npolys; i++)
    {
        if (polygons[i]->Degenerate) continue;
        PolygonRanges[j] = PolygonLines(polygons[i]);
        SetupPolygon(&PolygonList[j++], polygons[i]);
    }

    TopContext = {PolygonList, PolygonRanges, StencilBuffer, PrevIsShadowMask};

    RenderScanline(TopContext, 0, j);
    for (s32 y = 1; y < SplitLine; y++)
    {
        if (CancelFrame.load(std::memory_order_relaxed))
            break;

        RenderScanline(TopContext, y, j);
        ScanlineFinalPass(y-1);

        std::lock_guard lock(PublishMutex);
        TopReady = y;
        PublishLines();
    }

    TopHalfDone.store(true, std::memory_order_release);
    Platform::Semaphore_Post(Sema_TopHalfDone);

    Platform::Semaphore_Wait(Sema_SplitDone);

    // carry the rasterizer state over to the next frame as if one thread had drawn it all
    if (SplitTookTopStencil)
    {
        memcpy(StencilBuffer, SplitStencilBuffer, sizeof(StencilBuffer));
        PrevIsShadowMask = SplitContext.PrevIsShadowMask;
    }
    else
    {
        for (int parity = 0; parity < 2; parity++)
        {
            if (SplitStencilValid[parity])
                memcpy(&StencilBuffer[256 * parity], &SplitStencilBuffer[256 * parity], 256);
        }
        // the split thread started from the top half's final shadow mask state, so this is right
        // even if it didn't draw anything
        PrevIsShadowMask = SplitContext.PrevIsShadowMask;
    }
}

void SoftRenderer3D::PublishLines()
{
    // PublishMutex is held
    s32 ready = BoundaryReady ? (SplitLine + 1 + BottomReady) : TopReady;
    if (ready > LinesPublished)
    {
        Platform::Semaphore_Post(Sema_ScanlineCount, ready - LinesPublished);
        LinesPublished = ready;
    }
}

static bool PolygonCoversLine(const Polygon* polygon, s32 y)
{
    return y >= polygon->YTop && (y < polygon->YBottom || (y == polygon->YTop && polygon->YBottom == polygon->YTop));
}

bool SoftRenderer3D::BottomHalfNeedsTopStencil(bool& prevIsShadowMask, bool* valid) const
{
    // the shadow mask state at the start of SplitLine comes from the last polygon drawn before it:
    // the last one in the list among those reaching furthest down above SplitLine
    s32 lastLine = -1;
    bool hasShadows = false;
    prevIsShadowMask = SplitStartPrevIsShadowMask;
    for (int i = 0; i < SplitNumPolygons; i++)
    {
        const Polygon* polygon = SplitPolygons[i];
        if (polygon->Degenerate) continue;
        if (polygon->IsShadowMask || polygon->IsShadow) hasShadows = true;

        s32 bottom = (polygon->YBottom == polygon->YTop) ? polygon->YTop : polygon->YBottom - 1;
        if (bottom > SplitLine - 1) bottom = SplitLine - 1;
        if (polygon->YTop > bottom) continue;
        if (bottom >= lastLine)
        {
            lastLine = bottom;
            prevIsShadowMask = polygon->IsShadowMask;
        }
    }

    valid[0] = valid[1] = false;
    if (!hasShadows)
        return false;

    // follow the stencil buffer through the bottom half: it must be cleared
    // by a shadow mask before anything reads it or adds to it
    bool prev = prevIsShadowMask;
    for (s32 y = SplitLine; y < 192; y++)
    {
        for (int i = 0; i < SplitNumPolygons; i++)
        {
            const Polygon* polygon = SplitPolygons[i];
            if (polygon->Degenerate || !PolygonCoversLine(polygon, y)) continue;

            if (polygon->IsShadowMask)
            {
                if (!prev)
                    valid[y & 1] = true;
                else if (!valid[y & 1])
                    return true;
                prev = true;
            }
            else
            {
                if (polygon->IsShadow && !valid[y & 1])
                    return true;
                prev = false;
            }
        }
    }

    return false;
}

void SoftRenderer3D::RenderBottomHalf()
{
    bool prevIsShadowMask;
    SplitTookTopStencil = BottomHalfNeedsTopStencil(prevIsShadowMask, SplitStencilValid);

    int j = 0;
    for (int i = 0; i < SplitNumPolygons; i++)
    {
        Polygon* polygon = SplitPolygons[i];
        if (polygon->Degenerate) continue;

        SplitPolygonRanges[j] = PolygonLines(polygon);
        RendererPolygon* rp = &SplitPolygonList[j++];
        SetupPolygon(rp, polygon);

        // stepping the edges down to SplitLine gives the same state as setting them up there
        if (polygon->YTop < SplitLine && SplitLine < polygon->YBottom)
        {
            SetupPolygonLeftEdge(rp, SplitLine);
            SetupPolygonRightEdge(rp, SplitLine);
        }
    }

    if (SplitTookTopStencil)
    {
        // the bottom half depends on what the top half left in the stencil buffer
        Platform::Semaphore_Wait(Sema_TopHalfDone);
        memcpy(SplitStencilBuffer, StencilBuffer, sizeof(SplitStencilBuffer));
        prevIsShadowMask = TopContext.PrevIsShadowMask;
    }

    SplitContext = {SplitPolygonList, SplitPolygonRanges, SplitStencilBuffer, prevIsShadowMask};

    bool boundaryDone = false;
    auto finishBoundary = [&]() {
        // the top half's final pass for SplitLine-2 is done once TopHalfDone is set
        ScanlineFinalPass(SplitLine - 1);
        ScanlineFinalPass(SplitLine);
        boundaryDone = true;
        std::lock_guard lock(PublishMutex);
        BoundaryReady = true;
        PublishLines();
    };

    RenderScanline(SplitContext, SplitLine, j);
    for (s32 y = SplitLine + 1; y < 192; y++)
    {
        if (CancelFrame.load(std::memory_order_relaxed))
            break;

        RenderScanline(SplitContext, y, j);

        if (y - 1 > SplitLine)
        {
            ScanlineFinalPass(y - 1);
            std::lock_guard lock(PublishMutex);
            BottomReady = y - 1 - SplitLine;
            PublishLines();
        }

        if (!boundaryDone && TopHalfDone.load(std::memory_order_acquire))
            finishBoundary();
    }

    ScanlineFinalPass(191);
    {
        std::lock_guard lock(PublishMutex);
        BottomReady = 191 - SplitLine;
        PublishLines();
    }

    if (!boundaryDone)
    {
        if (!SplitTookTopStencil)
            Platform::Semaphore_Wait(Sema_TopHalfDone);
        finishBoundary();
    }
}

void SoftRenderer3D::SplitThreadFunc()
{
    for (;;)
    {
        Platform::Semaphore_Wait(Sema_SplitStart);
        if (!SplitThreadRunning) return;

        RenderBottomHalf();

        Platform::Semaphore_Post(Sema_SplitDone);
    }
}

void SoftRenderer3D::FinishRendering()
{
    if (RenderThreadRunning.load(std::memory_order_relaxed) && !GPU3D.AbortFrame)
        Platform::Semaphore_Wait(Sema_RenderDone);
}

bool SoftRenderer3D::RenderFrameEarly()
{
    EarlyFrame = false;
    if (!RenderThreadRunning.load(std::memory_order_relaxed) || GPU3D.AbortFrame)
        return false;

    // the render threads are idle (see FinishRendering), so the shadow state can be saved
    memcpy(EarlyStencilBuffer, StencilBuffer, sizeof(StencilBuffer));
    EarlyPrevIsShadowMask = PrevIsShadowMask;

    RenderFrame();
    EarlyFrame = true;
    return true;
}

void SoftRenderer3D::RenderFrame()
{
    auto textureDirty = GPU.VRAMDirty_Texture.DeriveState(GPU.VRAMMap_Texture, GPU);
    auto texPalDirty = GPU.VRAMDirty_TexPal.DeriveState(GPU.VRAMMap_TexPal, GPU);

    if (EarlyFrame)
    {
        EarlyFrame = false;

        // nothing the frame depends on changed since it was started
        if (!textureDirty && !texPalDirty)
            return;

        // stop the frame and start it over with the new texture memory
        CancelFrame = true;
        Platform::Semaphore_Wait(Sema_RenderDone);
        CancelFrame = false;
        Platform::Semaphore_Reset(Sema_ScanlineCount);

        memcpy(StencilBuffer, EarlyStencilBuffer, sizeof(StencilBuffer));
        PrevIsShadowMask = EarlyPrevIsShadowMask;

        GPU.MakeVRAMFlat_TextureCoherent(textureDirty);
        GPU.MakeVRAMFlat_TexPalCoherent(texPalDirty);

        // the buffers hold part of the cancelled frame now
        FrameIdentical = false;
        Platform::Semaphore_Post(Sema_RenderStart);
        return;
    }

    bool textureChanged = GPU.MakeVRAMFlat_TextureCoherent(textureDirty);
    bool texPalChanged = GPU.MakeVRAMFlat_TexPalCoherent(texPalDirty);

    FrameIdentical = !(textureChanged || texPalChanged) && GPU3D.RenderFrameIdentical;

    if (RenderThreadRunning.load(std::memory_order_relaxed))
    {
        // "Render thread, you're up! Get moving."
        Platform::Semaphore_Post(Sema_RenderStart);
    }
    else if (!FrameIdentical)
    {
        ClearBuffers();
        RenderPolygons(false, &GPU3D.RenderPolygonRAM[0], GPU3D.RenderNumPolygons);
    }
}

void SoftRenderer3D::RestartFrame()
{
    EarlyFrame = false;
    SetupRenderThread();
    EnableRenderThread();
}

void SoftRenderer3D::RenderThreadFunc()
{
    for (;;)
    {
        // Wait for a notice from the main thread to start rendering (or to stop entirely).
        Platform::Semaphore_Wait(Sema_RenderStart);
        if (!RenderThreadRunning) return;

        // Protect the GPU state from the main thread.
        // Some melonDS frontends (though not ours)
        // will repeatedly save or load states;
        // if they do so while the render thread is busy here,
        // the ensuing race conditions may cause a crash
        // (since some of the GPU state includes pointers).
        RenderThreadRendering = true;
        if (FrameIdentical)
        { // If no rendering is needed, just say we're done.
            Platform::Semaphore_Post(Sema_ScanlineCount, 192);
        }
        else
        {
            ClearBuffers();
            RenderPolygons(true, &GPU3D.RenderPolygonRAM[0], GPU3D.RenderNumPolygons);
        }

        // Tell the main thread that we're done rendering
        // and that it's safe to access the GPU state again.
        Platform::Semaphore_Post(Sema_RenderDone);

        RenderThreadRendering = false;
    }
}

u32* SoftRenderer3D::GetLine(int line)
{
    if (GPU3D.AbortFrame)
    {
        // TODO this isn't accurate
        memset(ScrolledLine, 0, sizeof(ScrolledLine));
        return ScrolledLine;
    }

    if (RenderThreadRunning.load(std::memory_order_relaxed))
    {
        if (line < 192)
            // We need a scanline, so let's wait for the render thread to finish it.
            // (both threads process scanlines from top-to-bottom,
            // so we don't need to wait for a specific row)
            Platform::Semaphore_Wait(Sema_ScanlineCount);
    }

    u32* rawline = &ColorBuffer[(line * ScanlineWidth) + FirstPixelOffset];
    u16 xpos = GPU3D.RenderXPos;
    if (xpos == 0)
        return rawline;

    // apply X scroll

    if (xpos & 0x100)
    {
        int i = 0, j = xpos;
        for (; j < 512; i++, j++)
            ScrolledLine[i] = 0;
        for (j = 0; i < 256; i++, j++)
            ScrolledLine[i] = rawline[j];
    }
    else
    {
        int i = 0, j = xpos;
        for (; j < 256; i++, j++)
            ScrolledLine[i] = rawline[j];
        for (; i < 256; i++)
            ScrolledLine[i] = 0;
    }

    return ScrolledLine;
}

}
