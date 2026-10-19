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

#ifndef OPENGLSUPPORT_H
#define OPENGLSUPPORT_H

#include <stdio.h>
#include <string.h>

#include "Platform.h"
#include "PlatformOGL.h"

#ifndef GL_CLAMP_TO_BORDER
#define GL_CLAMP_TO_BORDER 0x812D // OpenGL ES 3.2
#endif

namespace melonDS::OpenGL
{

// The DS's 15-bit color format (BGR555, bit 15 = alpha) matches GL_UNSIGNED_SHORT_1_5_5_5_REV,
// which OpenGL ES doesn't have. There, textures of this format use GL_UNSIGNED_SHORT_5_5_5_1
// and the pixels are converted on upload and on readback.
#ifdef OGLRENDERER_GLES
constexpr GLenum PixelType1555 = GL_UNSIGNED_SHORT_5_5_5_1;
#else
constexpr GLenum PixelType1555 = GL_UNSIGNED_SHORT_1_5_5_5_REV;
#endif

// Returns pixels in the layout PixelType1555 expects;
// on OpenGL ES this is a converted copy that stays valid until the next call.
const void* Prepare1555Upload(const u16* src, size_t count);

// Reads back an RGB5_A1 framebuffer region into the DS's 15-bit color format
void ReadPixels1555(GLint x, GLint y, GLsizei width, GLsizei height, u16* dst);

void LoadShaderCache();
void SaveShaderCache();

struct AttributeTarget
{
    const char* Name;
    u32 Location;
};


bool CompileVertexFragmentProgram(GLuint& result,
    const std::string& vs, const std::string& fs,
    const std::string& name,
    const std::initializer_list<AttributeTarget>& vertexInAttrs,
    const std::initializer_list<AttributeTarget>& fragmentOutAttrs);

bool CompileComputeProgram(GLuint& result, const std::string& source, const std::string& name);

}

#endif // OPENGLSUPPORT_H
