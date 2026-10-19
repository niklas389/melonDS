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

#include "OpenGLSupport.h"

#include <unordered_map>
#include <vector>

#include <assert.h>

#define XXH_STATIC_LINKING_ONLY
#include "xxhash/xxhash.h"

namespace melonDS
{

using Platform::Log;
using Platform::LogLevel;

namespace OpenGL
{

struct ShaderCacheEntry
{
    u32 Length;
    u8* Data;
    u32 BinaryFormat;

    ShaderCacheEntry(u8* data, u32 length, u32 binaryFmt)
        : Length(length), Data(data), BinaryFormat(binaryFmt)
    {
        assert(data != nullptr);
    }

    ShaderCacheEntry(const ShaderCacheEntry&) = delete;
    ShaderCacheEntry(ShaderCacheEntry&& other)
    {
        Data = other.Data;
        Length = other.Length;
        BinaryFormat = other.BinaryFormat;

        other.Data = nullptr;
        other.Length = 0;
        other.BinaryFormat = 0;
    }

    ~ShaderCacheEntry()
    {
        if (Data) // check whether it was moved
            delete[] Data;
    }
};

std::unordered_map<u64, ShaderCacheEntry> ShaderCache;
std::vector<u64> NewShaders;

constexpr u32 ShaderCacheMagic = 0x11CAC4E1;
constexpr u32 ShaderCacheVersion = 1;

void LoadShaderCache()
{
    // for now the shader cache only contains only compute shaders
    // because they take the longest to compile
    Platform::FileHandle* file = Platform::OpenLocalFile("shadercache", Platform::FileMode::Read);
    if (file == nullptr)
    {
        Log(LogLevel::Error, "Could not find shader cache\n");
        return;
    }

    u32 magic, version, numPrograms;
    if (Platform::FileRead(&magic, 4, 1, file) != 1 || magic != ShaderCacheMagic)
    {
        Log(LogLevel::Error, "Shader cache file has invalid magic\n");
        goto fileInvalid;
    }

    if (Platform::FileRead(&version, 4, 1, file) != 1 || version != ShaderCacheVersion)
    {
        Log(LogLevel::Error, "Shader cache file has bad version\n");
        goto fileInvalid;
    }

    if (Platform::FileRead(&numPrograms, 4, 1, file) != 1)
    {
        Log(LogLevel::Error, "Shader cache file invalid program count\n");
        goto fileInvalid;
    }

    // not the best approach, because once changes pile up
    // we read and overwrite the old files
    for (u32 i = 0; i < numPrograms; i++)
    {
        int error = 3;

        u32 length, binaryFormat;
        u64 sourceHash;
        error -= Platform::FileRead(&sourceHash, 8, 1, file);
        error -= Platform::FileRead(&length, 4, 1, file);
        error -= Platform::FileRead(&binaryFormat, 4, 1, file);

        if (error != 0)
        {
            Log(LogLevel::Error, "Invalid shader cache entry\n");
            goto fileInvalid;
        }

        u8* data = new u8[length];
        if (Platform::FileRead(data, length, 1, file) != 1)
        {
            Log(LogLevel::Error, "Could not read shader cache entry data\n");
            delete[] data;
            goto fileInvalid;
        }

        ShaderCache.erase(sourceHash);
        ShaderCache.emplace(sourceHash, ShaderCacheEntry(data, length, binaryFormat));
    }

fileInvalid:
    Platform::CloseFile(file);
}

void SaveShaderCache()
{
    Platform::FileHandle* file = Platform::OpenLocalFile("shadercache", Platform::FileMode::ReadWrite);

    if (file == nullptr)
    {
        Log(LogLevel::Error, "Could not open or create shader cache file\n");
        return;
    }

    int written = 3;
    u32 magic = ShaderCacheMagic, version = ShaderCacheVersion, numPrograms = ShaderCache.size();
    written -= Platform::FileWrite(&magic, 4, 1, file);
    written -= Platform::FileWrite(&version, 4, 1, file);
    written -= Platform::FileWrite(&numPrograms, 4, 1, file);

    if (written != 0)
    {
        Log(LogLevel::Error, "Could not write shader cache header\n");
        goto writeError;
    }

    Platform::FileSeek(file, 0, Platform::FileSeekOrigin::End);

    printf("new shaders %zu\n", NewShaders.size());

    for (u64 newShader : NewShaders)
    {
        int error = 4;
        auto it = ShaderCache.find(newShader);

        error -= Platform::FileWrite(&it->first, 8, 1, file);
        error -= Platform::FileWrite(&it->second.Length, 4, 1, file);
        error -= Platform::FileWrite(&it->second.BinaryFormat, 4, 1, file);
        error -= Platform::FileWrite(it->second.Data, it->second.Length, 1, file);

        if (error != 0)
        {
            Log(LogLevel::Error, "Could not insert new shader cache entry\n");
            goto writeError;
        }
    }

writeError:
    Platform::CloseFile(file);

    NewShaders.clear();
}

const void* Prepare1555Upload(const u16* src, size_t count)
{
#ifdef OGLRENDERER_GLES
    static std::vector<u16> converted;
    converted.resize(count);
    for (size_t i = 0; i < count; i++)
    {
        u16 c = src[i];
        converted[i] = ((c & 0x001F) << 11) | ((c & 0x03E0) << 1) | ((c & 0x7C00) >> 9) | (c >> 15);
    }
    return converted.data();
#else
    return src;
#endif
}

void ReadPixels1555(GLint x, GLint y, GLsizei width, GLsizei height, u16* dst)
{
#ifdef OGLRENDERER_GLES
    // OpenGL ES only guarantees GL_RGBA + GL_UNSIGNED_BYTE for glReadPixels
    static std::vector<u8> rgba;
    size_t count = (size_t)width * height;
    rgba.resize(count * 4);
    glReadPixels(x, y, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    for (size_t i = 0; i < count; i++)
    {
        const u8* p = &rgba[i * 4];
        dst[i] = (p[0] >> 3) | ((p[1] >> 3) << 5) | ((p[2] >> 3) << 10) | ((p[3] >> 7) << 15);
    }
#else
    glReadPixels(x, y, width, height, GL_RGBA, GL_UNSIGNED_SHORT_1_5_5_5_REV, dst);
#endif
}

#ifdef OGLRENDERER_GLES
// The shaders are written for GLSL 1.40; GLSL ES 3.00 is close enough
// once it gets its own version line, default precisions,
// and explicit locations for the fragment outputs (it has no glBindFragDataLocation).
static std::string ToGLSLES(const std::string& source, const std::initializer_list<AttributeTarget>& fragmentOutAttrs)
{
    std::string result = source;
    size_t version = result.find("#version");
    if (version != std::string::npos)
    {
        size_t end = result.find('\n', version);
        result.erase(version, end == std::string::npos ? std::string::npos : end - version);
    }

    for (const AttributeTarget& target : fragmentOutAttrs)
    {
        // Find "out <type> <name>;" and give it a location
        std::string name = target.Name;
        for (size_t pos = result.find(name); pos != std::string::npos; pos = result.find(name, pos + 1))
        {
            if (pos == 0 || (result[pos - 1] != ' ' && result[pos - 1] != '\t'))
                continue;

            size_t after = pos + name.size();
            size_t semicolon = result.find_first_not_of(" \t", after);
            if (semicolon == std::string::npos || result[semicolon] != ';')
                continue;

            size_t lineStart = result.rfind('\n', pos);
            lineStart = (lineStart == std::string::npos) ? 0 : lineStart + 1;
            size_t out = result.find_first_not_of(" \t", lineStart);
            if (out == std::string::npos || result.compare(out, 4, "out ") != 0 || out > pos)
                continue;

            std::string layout = "layout(location = " + std::to_string(target.Location) + ") ";
            result.insert(out, layout);
            break;
        }
    }

    return "#version 300 es\n"
           "precision highp float;\n"
           "precision highp int;\n"
           "precision highp sampler2D;\n"
           "precision highp sampler2DArray;\n"
           "precision highp usampler2D;\n"
           "precision highp usampler2DArray;\n"
           "precision highp isampler2D;\n"
           + result;
}
#endif

bool CompilerShader(GLuint& id, const std::string& source, const std::string& name, const std::string& type)
{
    int res;

    if (!glCreateShader)
    {
        Log(LogLevel::Error, "OpenGL: Cannot build shader program, OpenGL hasn't been loaded\n");
        return false;
    }

    const char* sourceC = source.c_str();
    int len = source.length();
    glShaderSource(id, 1, &sourceC, &len);

    glCompileShader(id);

    glGetShaderiv(id, GL_COMPILE_STATUS, &res);
    if (res != GL_TRUE)
    {
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &res);
        if (res < 1) res = 1024;
        char* log = new char[res+1];
        glGetShaderInfoLog(id, res+1, NULL, log);
        Log(LogLevel::Error, "OpenGL: failed to compile %s shader %s: %s\n", type.c_str(), name.c_str(), log);
        Log(LogLevel::Debug, "shader source:\n--\n%s\n--\n", source.c_str());
        delete[] log;

        glDeleteShader(id);

        return false;
    }

    return true;
}

bool LinkProgram(GLuint& result, GLuint* ids, int numIds)
{
    int res;

    if (!glLinkProgram)
    {
        Log(LogLevel::Error, "OpenGL: Cannot link shader program, OpenGL hasn't been loaded\n");
        return false;
    }

    for (int i = 0; i < numIds; i++)
    {
        glAttachShader(result, ids[i]);
    }

    glLinkProgram(result);

    for (int i = 0; i < numIds; i++)
        glDetachShader(result, ids[i]);

    glGetProgramiv(result, GL_LINK_STATUS, &res);
    if (res != GL_TRUE)
    {
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &res);
        if (res < 1) res = 1024;
        char* log = new char[res+1];
        glGetProgramInfoLog(result, res+1, NULL, log);
        Log(LogLevel::Error, "OpenGL: failed to link shader program: %s\n", log);
        delete[] log;

        return false;
    }

    return true;
}

bool CompileComputeProgram(GLuint& result, const std::string& source, const std::string& name)
{
    result = glCreateProgram();

    /*u64 sourceHash = XXH64(source.data(), source.size(), 0);
    auto it = ShaderCache.find(sourceHash);
    if (it != ShaderCache.end())
    {
        glProgramBinary(result, it->second.BinaryFormat, it->second.Data, it->second.Length);

        GLint linkStatus;
        glGetProgramiv(result, GL_LINK_STATUS, &linkStatus);
        if (linkStatus == GL_TRUE)
        {
            Log(LogLevel::Info, "Restored shader %s from cache\n", name.c_str());
            return true;
        }
        else
        {
        }
    }*/
    Log(LogLevel::Error, "Shader %s from cache was rejected\n", name.c_str());

    GLuint shader;
    bool linkingSucess = false;

    if (!glCreateShader || !glDeleteShader)
        goto error;

    shader = glCreateShader(GL_COMPUTE_SHADER);

    if (!CompilerShader(shader, source, name, "compute"))
        goto error;

    linkingSucess = LinkProgram(result, &shader, 1);

error:
    glDeleteShader(shader);

    if (!linkingSucess)
    {
        glDeleteProgram(result);
    }
    /*else
    {
        GLint length;
        GLenum format;
        glGetProgramiv(result, GL_PROGRAM_BINARY_LENGTH, &length);

        u8* buffer = new u8[length];
        glGetProgramBinary(result, length, nullptr, &format, buffer);

        ShaderCache.emplace(sourceHash, ShaderCacheEntry(buffer, length, format));
        NewShaders.push_back(sourceHash);
    }*/

    return linkingSucess;
}

bool CompileVertexFragmentProgram(GLuint& result,
    const std::string& vs, const std::string& fs,
    const std::string& name,
    const std::initializer_list<AttributeTarget>& vertexInAttrs,
    const std::initializer_list<AttributeTarget>& fragmentOutAttrs)
{
    GLuint shaders[2] =
    {
        glCreateShader(GL_VERTEX_SHADER),
        glCreateShader(GL_FRAGMENT_SHADER)
    };
    result = glCreateProgram();

    bool linkingSucess = false;

#ifdef OGLRENDERER_GLES
    if (!CompilerShader(shaders[0], ToGLSLES(vs, {}), name, "vertex"))
        goto error;

    if (!CompilerShader(shaders[1], ToGLSLES(fs, fragmentOutAttrs), name, "fragment"))
        goto error;
#else
    if (!CompilerShader(shaders[0], vs, name, "vertex"))
        goto error;

    if (!CompilerShader(shaders[1], fs, name, "fragment"))
        goto error;
#endif

    for (const AttributeTarget& target : vertexInAttrs)
    {
        glBindAttribLocation(result, target.Location, target.Name);
    }
#ifndef OGLRENDERER_GLES
    for (const AttributeTarget& target : fragmentOutAttrs)
    {
        glBindFragDataLocation(result, target.Location, target.Name);
    }
#endif

    linkingSucess = LinkProgram(result, shaders, 2);

error:
    glDeleteShader(shaders[1]);
    glDeleteShader(shaders[0]);

    if (!linkingSucess)
        glDeleteProgram(result);

    return linkingSucess;
}

}

}