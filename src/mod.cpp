// Core of the mod: D3D11 vtable hooks, TAA pass detection, DLSS evaluation.
//
// Ghost Recon Breakpoint (AnvilNext, DX11) records its frame on deferred contexts. The TAA pass
// (TemporalAA_* pixel shader: low-res colour/depth/motion -> full-res output) is therefore recorded
// on a worker thread. At that draw we:
//   1. capture the TAA constants (jitter, sizes) that the game wrote into its constant buffers,
//   2. split the deferred command list (FinishCommandList) right before the TAA draw,
//   3. record, instead of the TAA shader, a "resolve" draw that copies the DLSS output into the
//      TAA output render target (using the game's own fullscreen draw and state).
// When the game later submits that command list on the immediate context (ExecuteCommandList),
// we execute the prefix list, run input preparation + NGX DLSS on the immediate context, then let
// the game's command list (which starts with the resolve draw) execute.
#include "common.h"
#include "ngx.h"
#include "shaders.h"

// ------------------------------------------------------------------ vtable slots (verified against d3d11_1.h)
enum
{
    kDev_CreateRenderTargetView = 9,
    kDev_CreatePixelShader = 15,
    kDev_CreateComputeShader = 18,
    kDev_CreateDeferredContext = 27,
    kCtx_PSSetShader = 9,
    kCtx_DrawIndexed = 12,
    kCtx_Draw = 13,
    kCtx_Map = 14,
    kCtx_Unmap = 15,
    kCtx_DrawIndexedInstanced = 20,
    kCtx_DrawInstanced = 21,
    kCtx_Dispatch = 41,
    kCtx_RSSetViewports = 44,
    kCtx_CopySubresourceRegion = 46,
    kCtx_CopyResource = 47,
    kCtx_UpdateSubresource = 48,
    kCtx_ExecuteCommandList = 58,
    kCtx_ClearState = 110,
    kCtx_FinishCommandList = 114,
    kCtx1_UpdateSubresource1 = 116,
    kMaxSlots = 140,
};

typedef HRESULT(STDMETHODCALLTYPE* PFN_CreateComputeShader)(ID3D11Device*, const void*, SIZE_T, ID3D11ClassLinkage*,
                                                           ID3D11ComputeShader**);
typedef void(STDMETHODCALLTYPE* PFN_RSSetViewports)(ID3D11DeviceContext*, UINT, const D3D11_VIEWPORT*);
typedef void(STDMETHODCALLTYPE* PFN_Dispatch)(ID3D11DeviceContext*, UINT, UINT, UINT);
typedef HRESULT(STDMETHODCALLTYPE* PFN_CreateDeferredContext)(ID3D11Device*, UINT, ID3D11DeviceContext**);
typedef HRESULT(STDMETHODCALLTYPE* PFN_CreatePixelShader)(ID3D11Device*, const void*, SIZE_T, ID3D11ClassLinkage*,
                                                          ID3D11PixelShader**);
typedef void(STDMETHODCALLTYPE* PFN_PSSetShader)(ID3D11DeviceContext*, ID3D11PixelShader*, ID3D11ClassInstance* const*, UINT);
typedef void(STDMETHODCALLTYPE* PFN_Draw)(ID3D11DeviceContext*, UINT, UINT);
typedef void(STDMETHODCALLTYPE* PFN_DrawIndexed)(ID3D11DeviceContext*, UINT, UINT, INT);
typedef void(STDMETHODCALLTYPE* PFN_DrawInstanced)(ID3D11DeviceContext*, UINT, UINT, UINT, UINT);
typedef void(STDMETHODCALLTYPE* PFN_DrawIndexedInstanced)(ID3D11DeviceContext*, UINT, UINT, UINT, INT, UINT);
typedef HRESULT(STDMETHODCALLTYPE* PFN_Map)(ID3D11DeviceContext*, ID3D11Resource*, UINT, D3D11_MAP, UINT,
                                            D3D11_MAPPED_SUBRESOURCE*);
typedef void(STDMETHODCALLTYPE* PFN_Unmap)(ID3D11DeviceContext*, ID3D11Resource*, UINT);
typedef void(STDMETHODCALLTYPE* PFN_UpdateSubresource)(ID3D11DeviceContext*, ID3D11Resource*, UINT, const D3D11_BOX*,
                                                       const void*, UINT, UINT);
typedef void(STDMETHODCALLTYPE* PFN_UpdateSubresource1)(ID3D11DeviceContext*, ID3D11Resource*, UINT, const D3D11_BOX*,
                                                        const void*, UINT, UINT, UINT);
typedef void(STDMETHODCALLTYPE* PFN_CopySubresourceRegion)(ID3D11DeviceContext*, ID3D11Resource*, UINT, UINT, UINT, UINT,
                                                           ID3D11Resource*, UINT, const D3D11_BOX*);
typedef void(STDMETHODCALLTYPE* PFN_CopyResource)(ID3D11DeviceContext*, ID3D11Resource*, ID3D11Resource*);
typedef void(STDMETHODCALLTYPE* PFN_ExecuteCommandList)(ID3D11DeviceContext*, ID3D11CommandList*, BOOL);
typedef void(STDMETHODCALLTYPE* PFN_ClearState)(ID3D11DeviceContext*);
typedef HRESULT(STDMETHODCALLTYPE* PFN_FinishCommandList)(ID3D11DeviceContext*, BOOL, ID3D11CommandList**);

struct VtHooks
{
    void** vt;
    void* orig[kMaxSlots];
};
static VtHooks s_ctxHooks[16];
static volatile LONG s_ctxHookCount;
static SRWLOCK s_hookLock = SRWLOCK_INIT;
static void** s_devVt;
static PFN_CreatePixelShader o_CreatePixelShader;
static PFN_CreateDeferredContext o_CreateDeferredContext;
static PFN_CreateComputeShader o_CreateComputeShader;
static ID3D11Device* volatile s_lastHookedDev;
static void HookContextVtable(ID3D11DeviceContext* ctx);
static void HookOneVtable(ID3D11DeviceContext* ctx, const char* iface);
static void STDMETHODCALLTYPE Hk_ExecuteCommandList(ID3D11DeviceContext* ctx, ID3D11CommandList* cl, BOOL restore);

static void LogUnknownCtxVtable(void** vt);
static inline VtHooks* HooksFor(ID3D11DeviceContext* ctx)
{
    void** vt = *(void***)ctx;
    LONG n = s_ctxHookCount;
    for (LONG i = n - 1; i >= 0; i--)
        if (s_ctxHooks[i].vt == vt)
            return &s_ctxHooks[i];
    // a vtable another layer copied from one we had patched: the originals of the most recently
    // hooked vtable are the best match (d3d11's context vtables point at the same functions)
    LogUnknownCtxVtable(vt);
    return n > 0 ? &s_ctxHooks[n - 1] : &s_ctxHooks[0];
}
#define ORIG(h, idx, T) ((T)(h)->orig[idx])

// ------------------------------------------------------------------ global state
static ID3D11PixelShader* volatile s_taaPS[64];
static volatile LONG s_taaCount;
static volatile DWORD s_modThread; // thread currently running our own work on the immediate context

static bool s_userEnabled = true; // toggled by hotkey
static ID3D11ShaderResourceView* volatile s_shadowMaskSrv; // sun shadow mask (debug view 6)
static volatile bool s_fatal;     // permanent failure: leave the game's TAA alone
static volatile LONG64 s_taaFrames, s_dlssFrames, s_fallbackFrames, s_jobFails;
static volatile uint64_t s_lastEvalQpc, s_lastTaaQpc;
static volatile UINT s_stLowW, s_stLowH, s_stFullW, s_stFullH;

// ------------------------------------------------------------------ per-context info
struct CtxInfo
{
    ID3D11DeviceContext* ctx;
    ID3D11PixelShader* ps;
    int type;       // -1 unknown, 0 immediate, 1 deferred
    int splitState; // deferred: list was split, NO_OVERWRITE maps need a DISCARD first
    ID3D11Resource* discarded[64];
    int discardedCount;
    struct Job* pending[4]; // jobs recorded since the last FinishCommandList
    int pendingCount;
};
static CtxInfo s_ctx[64];
static volatile LONG s_ctxCount;
static SRWLOCK s_ctxLock = SRWLOCK_INIT;

static CtxInfo* GetCtx(ID3D11DeviceContext* ctx)
{
    LONG n = s_ctxCount;
    for (LONG i = 0; i < n; i++)
        if (s_ctx[i].ctx == ctx)
            return &s_ctx[i];
    AcquireSRWLockExclusive(&s_ctxLock);
    CtxInfo* e = nullptr;
    n = s_ctxCount;
    for (LONG i = 0; i < n; i++)
        if (s_ctx[i].ctx == ctx)
            e = &s_ctx[i];
    if (!e)
    {
        e = n < 64 ? &s_ctx[n] : &s_ctx[63];
        memset(e, 0, sizeof(*e));
        e->type = -1;
        e->ctx = ctx;
        MemoryBarrier();
        if (n < 64)
            s_ctxCount = n + 1;
    }
    ReleaseSRWLockExclusive(&s_ctxLock);
    return e;
}

static inline bool IsTaaPS(ID3D11PixelShader* ps)
{
    if (!ps)
        return false;
    LONG n = s_taaCount;
    for (LONG i = 0; i < n; i++)
        if (s_taaPS[i] == ps)
            return true;
    return false;
}

static bool MemFind(const void* hay, size_t hn, const char* needle)
{
    size_t nn = strlen(needle);
    const uint8_t* h = (const uint8_t*)hay;
    if (hn < nn)
        return false;
    for (size_t i = 0; i + nn <= hn; i++)
        if (h[i] == (uint8_t)needle[0] && memcmp(h + i, needle, nn) == 0)
            return true;
    return false;
}

// ------------------------------------------------------------------ constant buffer shadowing
// DLSS needs this frame's jitter and resolution on the CPU, so we capture what the game writes
// into the constant buffers bound to the TAA pass. Data is tracked per (buffer, context) because
// deferred contexts on different threads rename the same dynamic buffer independently.
enum
{
    kMaxWatch = 64,
    kMaxShadow = 128,
    kSmallCB = 65536,
    kShadowCap = 1 << 20,
};
struct Watch
{
    ID3D11Resource* res;
    UINT byteWidth;
};
static Watch s_watch[kMaxWatch];
static volatile LONG s_watchCount;

struct Shadow
{
    ID3D11Resource* res;
    ID3D11DeviceContext* ctx;
    UINT dataBytes;
    uint8_t* data;
    void* lastPtr;
    int method; // 1 data, 2 lastPtr (ring buffer)
    LONG64 seq;
};
static Shadow s_shadow[kMaxShadow];
static LONG s_shadowCount;
static LONG64 s_shadowSeq;
static SRWLOCK s_shadowLock = SRWLOCK_INIT;
static volatile LONG64 s_cbUpdates, s_cbGpuCopies;

struct PendingMap
{
    ID3D11DeviceContext* ctx;
    ID3D11Resource* res;
    UINT sub;
    void* ptr;
    UINT bytes;
};
static PendingMap s_pending[64];
static volatile LONG s_pendingCount;
static SRWLOCK s_pendingLock = SRWLOCK_INIT;

static inline int FindWatch(ID3D11Resource* r)
{
    LONG n = s_watchCount;
    for (LONG i = 0; i < n; i++)
        if (s_watch[i].res == r)
            return i;
    return -1;
}

static void WatchBuffer(ID3D11Buffer* b)
{
    if (FindWatch(b) >= 0)
        return;
    AcquireSRWLockExclusive(&s_shadowLock);
    if (FindWatch(b) < 0 && s_watchCount < kMaxWatch)
    {
        D3D11_BUFFER_DESC d;
        b->GetDesc(&d);
        s_watch[s_watchCount].byteWidth = d.ByteWidth;
        MemoryBarrier();
        s_watch[s_watchCount].res = b;
        s_watchCount++;
        LOGI("Watching constant buffer %p: ByteWidth=%u Usage=%u CPUAccess=0x%X Bind=0x%X", b, d.ByteWidth, d.Usage,
             d.CPUAccessFlags, d.BindFlags);
    }
    ReleaseSRWLockExclusive(&s_shadowLock);
}

// caller holds s_shadowLock exclusively
static Shadow* ShadowEntry(ID3D11Resource* r, ID3D11DeviceContext* ctx, UINT byteWidth)
{
    for (LONG i = 0; i < s_shadowCount; i++)
        if (s_shadow[i].res == r && s_shadow[i].ctx == ctx)
            return &s_shadow[i];
    Shadow* s;
    if (s_shadowCount < kMaxShadow)
        s = &s_shadow[s_shadowCount++];
    else
    {
        s = &s_shadow[0];
        for (LONG i = 1; i < kMaxShadow; i++)
            if (s_shadow[i].seq < s->seq)
                s = &s_shadow[i];
    }
    UINT want = byteWidth < kShadowCap ? byteWidth : kShadowCap;
    if (!s->data || s->dataBytes < want)
    {
        if (s->data)
            HeapFree(GetProcessHeap(), 0, s->data);
        s->data = (uint8_t*)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, want);
        s->dataBytes = s->data ? want : 0;
    }
    s->res = r;
    s->ctx = ctx;
    s->method = 0;
    s->lastPtr = nullptr;
    s->seq = 0;
    return s;
}

static void ShadowWrite(ID3D11Resource* r, ID3D11DeviceContext* ctx, UINT byteWidth, UINT off, const void* src, UINT bytes)
{
    AcquireSRWLockExclusive(&s_shadowLock);
    Shadow* s = ShadowEntry(r, ctx, byteWidth);
    if (s->data && off < s->dataBytes)
    {
        if (off + bytes > s->dataBytes)
            bytes = s->dataBytes - off;
        memcpy(s->data + off, src, bytes);
        s->method = 1;
        s->seq = ++s_shadowSeq;
    }
    ReleaseSRWLockExclusive(&s_shadowLock);
    InterlockedIncrement64(&s_cbUpdates);
}

static void ShadowPointer(ID3D11Resource* r, ID3D11DeviceContext* ctx, UINT byteWidth, void* p)
{
    AcquireSRWLockExclusive(&s_shadowLock);
    Shadow* s = ShadowEntry(r, ctx, byteWidth);
    s->lastPtr = p;
    s->method = 2;
    s->seq = ++s_shadowSeq;
    ReleaseSRWLockExclusive(&s_shadowLock);
    InterlockedIncrement64(&s_cbUpdates);
}

static bool Readable(const void* p, size_t n)
{
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery(p, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT)
        return false;
    if (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD))
        return false;
    if (!(mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE)))
        return false;
    return (const uint8_t*)p + n <= (const uint8_t*)mbi.BaseAddress + mbi.RegionSize;
}

// Reads the data the game last wrote to buffer r from context ctx (falls back to any context).
static bool ReadShadow(ID3D11Resource* r, ID3D11DeviceContext* ctx, UINT off, void* out, UINT bytes, bool* sameCtx)
{
    bool ok = false;
    AcquireSRWLockShared(&s_shadowLock);
    Shadow* best = nullptr;
    for (LONG i = 0; i < s_shadowCount; i++)
    {
        Shadow& s = s_shadow[i];
        if (s.res != r || !s.method)
            continue;
        if (s.ctx == ctx)
        {
            best = &s;
            break;
        }
        if (!best || s.seq > best->seq)
            best = &s;
    }
    if (best)
    {
        if (best->method == 2 && best->lastPtr && Readable((uint8_t*)best->lastPtr + off, bytes))
        {
            memcpy(out, (uint8_t*)best->lastPtr + off, bytes);
            ok = true;
        }
        else if (best->method == 1 && off + bytes <= best->dataBytes)
        {
            memcpy(out, best->data + off, bytes);
            ok = true;
        }
        if (sameCtx)
            *sameCtx = best->ctx == ctx;
    }
    ReleaseSRWLockShared(&s_shadowLock);
    return ok;
}

// ------------------------------------------------------------------ TAA constants
#pragma pack(push, 4)
struct TaaConstants
{
    float reproj[16];
    uint32_t isolation[4];
    float fullResSize[2];
    float fullResSizeInv[2];
    float lowResSize[2];
    float lowResSizeInv[2];
    float subPixelJitter[2];
    float rejMax, rejMaxInv;
    float noiseOffsetScale[3];
    float minLiveColorNear;
    float minLiveColorMisc[4];
    float maxViewDepthParams[4];
    uint32_t hdrEnabled;
    float hdrNormalizeSrgb;
    float hdrDenormalizeSrgb;
    uint32_t noMotionVector;
};
#pragma pack(pop)
static_assert(sizeof(TaaConstants) == 192, "TAA constant buffer layout");
enum
{
    kPassBytes = 864 // cb_g_Pass up to the end of m_PrevViewTranslation
};

// ------------------------------------------------------------------ DLSS views and jobs
struct View
{
    bool used;
    LONG64 lastUse;
    UINT lowW, lowH, fullW, fullH;
    int flags, quality, preset;
    LONG gen; // bumped whenever resources are recreated
    NVSDK_NGX_Handle* feature;
    NVSDK_NGX_Parameter* params;
    bool reset;
    ID3D11Texture2D *texColor, *texMV, *texDepth, *texBias, *texOut;
    ID3D11UnorderedAccessView *uavColor, *uavMV, *uavDepth, *uavBias;
    ID3D11ShaderResourceView *srvOut, *srvMV, *srvDepth;
    ID3D11Texture2D* texPrev;
    ID3D11ShaderResourceView* srvPrev;
    ID3D11ShaderResourceView* srvBias;
    ID3D11Texture2D* texMVHi;
    ID3D11UnorderedAccessView* uavMVHi;
    uint64_t evalCount;
    uint64_t lastQpc;
};
static View s_views[4];
static SRWLOCK s_viewLock = SRWLOCK_INIT;

struct Job
{
    View* view;
    LONG gen;
    ID3D11CommandList* prefix; // deferred: commands recorded before the TAA draw
    ID3D11ShaderResourceView *color, *motion, *depth, *resp;
    TaaConstants tc;
    uint8_t pass[kPassBytes];
    bool reset;
    bool hdr;
    LONG64 taaIndex;
    uint64_t qpc;
};

// command list -> jobs that must run before it
struct ClJobs
{
    ID3D11CommandList* cl; // AddRef'd
    Job* jobs[4];
    int count;
    uint64_t qpc;
};
static ClJobs s_clJobs[32];
static SRWLOCK s_clLock = SRWLOCK_INIT;

static void FreeJob(Job* j)
{
    if (!j)
        return;
    SafeRelease(j->prefix);
    SafeRelease(j->color);
    SafeRelease(j->motion);
    SafeRelease(j->depth);
    SafeRelease(j->resp);
    HeapFree(GetProcessHeap(), 0, j);
}

// ------------------------------------------------------------------ GPU resources of the mod
static ID3D11Device* s_dev;
static ID3D11Device1* s_dev1;
static ID3DDeviceContextState* s_ctxState;
static ID3D11ComputeShader* s_prepCS;
static ID3D11ComputeShader* s_dilateCS;
static ID3DBlob* s_blobDilate;
static ID3D11PixelShader* s_resolvePS;
static ID3D11Buffer* s_modCB_CS;
static ID3D11Buffer* s_modCB_PS;
static ID3D11Buffer* s_taaCB;
static ID3D11Buffer* s_passCB;
static ID3D11Buffer* s_infoBuf;
static ID3D11UnorderedAccessView* s_infoUAV;
static ID3D11Buffer* s_infoStaging[3];
static int s_infoWrite, s_infoPending;
static ID3D11Texture2D* s_dummyTex;
static ID3D11ShaderResourceView* s_dummySRV;
static ID3DBlob* s_blobCS;
static ID3DBlob* s_blobPS;
static volatile LONG s_gpuInitState; // 0 not started, 1 ok, 2 failed
static SRWLOCK s_initLock = SRWLOCK_INIT;
static bool s_ngxInitDone;

static volatile bool s_camValid;
static float s_camRev = 1.0f, s_camNear = 0.1f, s_camFar = 10000.0f, s_camP11 = 1.0f, s_camP00 = 1.0f;

typedef HRESULT(WINAPI* PFN_D3DCompile)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*, LPCSTR, LPCSTR,
                                       UINT, UINT, ID3DBlob**, ID3DBlob**);

static ID3DBlob* CompileHlsl(PFN_D3DCompile compile, const char* body, const char* target, const char* name)
{
    size_t a = strlen(kCommonHlsl), b = strlen(body);
    char* src = (char*)HeapAlloc(GetProcessHeap(), 0, a + b + 1);
    memcpy(src, kCommonHlsl, a);
    memcpy(src + a, body, b);
    src[a + b] = 0;
    ID3DBlob *code = nullptr, *err = nullptr;
    HRESULT hr = compile(src, a + b, name, nullptr, nullptr, "main", target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err);
    HeapFree(GetProcessHeap(), 0, src);
    if (FAILED(hr))
    {
        LOGE("Shader %s failed to compile (0x%08lX): %s", name, hr, err ? (const char*)err->GetBufferPointer() : "?");
        SafeRelease(err);
        SafeRelease(code);
        return nullptr;
    }
    SafeRelease(err);
    return code;
}

static bool CompileShaders()
{
    if (s_blobCS && s_blobPS)
        return true;
    wchar_t p[MAX_PATH];
    lstrcpyW(p, g_modDir);
    lstrcatW(p, L"d3dcompiler_47.dll");
    HMODULE m = LoadLibraryW(p);
    if (!m)
        m = LoadLibraryW(L"d3dcompiler_47.dll");
    if (!m)
    {
        LOGE("d3dcompiler_47.dll not found");
        return false;
    }
    PFN_D3DCompile compile = (PFN_D3DCompile)GetProcAddress(m, "D3DCompile");
    if (!compile)
        return false;
    if (!s_blobCS)
        s_blobCS = CompileHlsl(compile, kPrepCS, "cs_5_0", "GRB_DLSS_Prep");
    if (!s_blobDilate)
        s_blobDilate = CompileHlsl(compile, kDilateCS, "cs_5_0", "GRB_DLSS_DilateMV");
    if (!s_blobPS)
        s_blobPS = CompileHlsl(compile, kResolvePS, "ps_5_0", "GRB_DLSS_Resolve");
    LOGI("Shaders compiled: prep=%s resolve=%s", s_blobCS ? "ok" : "FAILED", s_blobPS ? "ok" : "FAILED");
    return s_blobCS && s_blobPS;
}

static ID3D11Buffer* MakeCB(UINT bytes)
{
    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth = (bytes + 15) & ~15u;
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    ID3D11Buffer* b = nullptr;
    s_dev->CreateBuffer(&bd, nullptr, &b);
    return b;
}

static bool InitGpuLocked(ID3D11DeviceContext* ctx)
{
    ctx->GetDevice(&s_dev);
    if (!s_dev)
        return false;
    s_dev->QueryInterface(__uuidof(ID3D11Device1), (void**)&s_dev1);
    if (!s_dev1)
    {
        LOGE("ID3D11Device1 not available");
        return false;
    }
    D3D_FEATURE_LEVEL fl = s_dev->GetFeatureLevel();
    UINT devFlags = s_dev->GetCreationFlags();
    LOGI("Game device %p: feature level 0x%X, creation flags 0x%X", s_dev, fl, devFlags);
    HRESULT hr = s_dev1->CreateDeviceContextState(
        (devFlags & D3D11_CREATE_DEVICE_SINGLETHREADED) ? D3D11_1_CREATE_DEVICE_CONTEXT_STATE_SINGLETHREADED : 0, &fl, 1,
        D3D11_SDK_VERSION, __uuidof(ID3D11Device1), nullptr, &s_ctxState);
    if (FAILED(hr))
    {
        LOGE("CreateDeviceContextState failed 0x%08lX", hr);
        return false;
    }
    if (!CompileShaders())
        return false;
    hr = s_dev->CreateComputeShader(s_blobCS->GetBufferPointer(), s_blobCS->GetBufferSize(), nullptr, &s_prepCS);
    if (FAILED(hr))
    {
        LOGE("CreateComputeShader failed 0x%08lX", hr);
        return false;
    }
    if (s_blobDilate)
        s_dev->CreateComputeShader(s_blobDilate->GetBufferPointer(), s_blobDilate->GetBufferSize(), nullptr, &s_dilateCS);
    if (!s_dilateCS)
        LOGE("Dilated motion vector shader unavailable - using render-resolution motion vectors");
    hr = s_dev->CreatePixelShader(s_blobPS->GetBufferPointer(), s_blobPS->GetBufferSize(), nullptr, &s_resolvePS);
    if (FAILED(hr))
    {
        LOGE("CreatePixelShader failed 0x%08lX", hr);
        return false;
    }
    s_modCB_CS = MakeCB(16);
    s_modCB_PS = MakeCB(16);
    s_taaCB = MakeCB(sizeof(TaaConstants));
    s_passCB = MakeCB(kPassBytes);
    if (!s_modCB_CS || !s_modCB_PS || !s_taaCB || !s_passCB)
        return false;

    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth = 32 * 4;
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    s_dev->CreateBuffer(&bd, nullptr, &s_infoBuf);
    if (s_infoBuf)
    {
        D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.Format = DXGI_FORMAT_R32_FLOAT;
        ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        ud.Buffer.NumElements = 32;
        s_dev->CreateUnorderedAccessView(s_infoBuf, &ud, &s_infoUAV);
    }
    bd.BindFlags = 0;
    bd.Usage = D3D11_USAGE_STAGING;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    for (int i = 0; i < 3; i++)
        s_dev->CreateBuffer(&bd, nullptr, &s_infoStaging[i]);

    D3D11_TEXTURE2D_DESC td = {};
    td.Width = td.Height = 1;
    td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    uint32_t zero = 0;
    D3D11_SUBRESOURCE_DATA sd = {&zero, 4, 4};
    s_dev->CreateTexture2D(&td, &sd, &s_dummyTex);
    if (s_dummyTex)
        s_dev->CreateShaderResourceView(s_dummyTex, nullptr, &s_dummySRV);
    LOGI("GPU resources ready");
    return true;
}

static bool InitGpu(ID3D11DeviceContext* ctx)
{
    LONG st = s_gpuInitState;
    if (st)
        return st == 1;
    AcquireSRWLockExclusive(&s_initLock);
    if (!s_gpuInitState)
        s_gpuInitState = InitGpuLocked(ctx) ? 1 : 2;
    ReleaseSRWLockExclusive(&s_initLock);
    return s_gpuInitState == 1;
}

static void ReleaseViewResources(View& v)
{
    if (v.feature && g_ngx.ReleaseFeature)
    {
        NVSDK_NGX_Result r = g_ngx.ReleaseFeature(v.feature);
        LOGI("DLSS feature released (0x%08X)", r);
    }
    v.feature = nullptr;
    SafeRelease(v.uavColor);
    SafeRelease(v.uavMV);
    SafeRelease(v.uavDepth);
    SafeRelease(v.uavBias);
    SafeRelease(v.srvOut);
    SafeRelease(v.srvMV);
    SafeRelease(v.srvDepth);
    SafeRelease(v.srvPrev);
    SafeRelease(v.srvBias);
    SafeRelease(v.texPrev);
    SafeRelease(v.uavMVHi);
    SafeRelease(v.texMVHi);
    SafeRelease(v.texColor);
    SafeRelease(v.texMV);
    SafeRelease(v.texDepth);
    SafeRelease(v.texBias);
    SafeRelease(v.texOut);
}

static ID3D11Texture2D* MakeTex(UINT w, UINT h, DXGI_FORMAT f, UINT bind)
{
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = w;
    td.Height = h;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = f;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = bind;
    ID3D11Texture2D* t = nullptr;
    HRESULT hr = s_dev->CreateTexture2D(&td, nullptr, &t);
    if (FAILED(hr))
        LOGE("CreateTexture2D %ux%u fmt %d failed 0x%08lX", w, h, f, hr);
    return t;
}

static bool CreateViewResources(View& v)
{
    const UINT rw = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    v.texColor = MakeTex(v.lowW, v.lowH, DXGI_FORMAT_R16G16B16A16_FLOAT, rw);
    v.texMV = MakeTex(v.lowW, v.lowH, DXGI_FORMAT_R16G16_FLOAT, rw);
    v.texDepth = MakeTex(v.lowW, v.lowH, DXGI_FORMAT_R32_FLOAT, rw);
    v.texBias = MakeTex(v.lowW, v.lowH, DXGI_FORMAT_R8_UNORM, rw);
    v.texOut = MakeTex(v.fullW, v.fullH, DXGI_FORMAT_R16G16B16A16_FLOAT, rw);
    if (!v.texColor || !v.texMV || !v.texDepth || !v.texBias || !v.texOut)
        return false;
    s_dev->CreateUnorderedAccessView(v.texColor, nullptr, &v.uavColor);
    s_dev->CreateUnorderedAccessView(v.texMV, nullptr, &v.uavMV);
    s_dev->CreateUnorderedAccessView(v.texDepth, nullptr, &v.uavDepth);
    s_dev->CreateUnorderedAccessView(v.texBias, nullptr, &v.uavBias);
    s_dev->CreateShaderResourceView(v.texOut, nullptr, &v.srvOut);
    s_dev->CreateShaderResourceView(v.texMV, nullptr, &v.srvMV);
    s_dev->CreateShaderResourceView(v.texDepth, nullptr, &v.srvDepth);
    if (!(v.flags & NGX_DLSS_MVLowRes))
    {
        v.texMVHi = MakeTex(v.fullW, v.fullH, DXGI_FORMAT_R16G16_FLOAT, rw);
        if (v.texMVHi)
            s_dev->CreateUnorderedAccessView(v.texMVHi, nullptr, &v.uavMVHi);
        if (!v.uavMVHi)
            return false;
    }
    if (g_cfg.devCommands)
    {
        v.texPrev = MakeTex(v.lowW, v.lowH, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D11_BIND_SHADER_RESOURCE);
        s_dev->CreateShaderResourceView(v.texBias, nullptr, &v.srvBias);
        if (v.texPrev)
            s_dev->CreateShaderResourceView(v.texPrev, nullptr, &v.srvPrev);
    }
    return v.uavColor && v.uavMV && v.uavDepth && v.uavBias && v.srvOut && v.srvMV && v.srvDepth;
}

static int PickQuality(UINT low, UINT full)
{
    if (g_cfg.qualityOverride >= 0)
        return g_cfg.qualityOverride;
    float r = (float)low / (float)full;
    if (r >= 0.95f)
        return NGX_PQ_DLAA;
    if (r >= 0.72f)
        return NGX_PQ_UltraQuality;
    if (r >= 0.62f)
        return NGX_PQ_MaxQuality;
    if (r >= 0.54f)
        return NGX_PQ_Balanced;
    if (r >= 0.42f)
        return NGX_PQ_MaxPerf;
    return NGX_PQ_UltraPerformance;
}

// caller holds s_viewLock exclusively
static View* GetViewLocked(UINT lowW, UINT lowH, UINT fullW, UINT fullH, int flags)
{
    View* v = nullptr;
    for (View& x : s_views)
        if (x.used && x.fullW == fullW && x.fullH == fullH)
            v = &x;
    if (!v)
    {
        for (View& x : s_views)
            if (!x.used)
            {
                v = &x;
                break;
            }
        if (!v)
        {
            v = &s_views[0];
            for (View& x : s_views)
                if (x.lastUse < v->lastUse)
                    v = &x;
            ReleaseViewResources(*v);
        }
        v->used = true;
        v->lowW = v->lowH = v->fullW = v->fullH = 0;
        LOGI("New DLSS view slot %d for %ux%u output", (int)(v - s_views), fullW, fullH);
    }
    int quality = PickQuality(lowW, fullW);
    if (v->lowW != lowW || v->lowH != lowH || v->fullW != fullW || v->fullH != fullH || v->flags != flags ||
        v->quality != quality || v->preset != g_cfg.preset)
    {
        if (v->lowW)
            LOGI("DLSS view %d reconfigure: %ux%u->%ux%u flags 0x%X  =>  %ux%u->%ux%u flags 0x%X", (int)(v - s_views),
                 v->lowW, v->lowH, v->fullW, v->fullH, v->flags, lowW, lowH, fullW, fullH, flags);
        // NGX feature release must happen on the immediate context's thread: keep the handle for RunJob
        NVSDK_NGX_Handle* oldFeature = v->feature;
        v->feature = nullptr;
        ReleaseViewResources(*v);
        v->feature = nullptr;
        if (oldFeature && g_ngx.ReleaseFeature)
            g_ngx.ReleaseFeature(oldFeature);
        v->lowW = lowW;
        v->lowH = lowH;
        v->fullW = fullW;
        v->fullH = fullH;
        v->flags = flags;
        v->quality = quality;
        v->preset = g_cfg.preset;
        v->gen++;
        v->evalCount = 0;
        if (!CreateViewResources(*v))
        {
            ReleaseViewResources(*v);
            v->lowW = 0;
            return nullptr;
        }
    }
    v->lastUse = s_taaFrames;
    return v;
}

static bool CreateFeature(ID3D11DeviceContext* imm, View& v)
{
    if (!v.params)
    {
        NVSDK_NGX_Result r = g_ngx.AllocateParameters(&v.params);
        if (!NGX_OK(r) || !v.params)
        {
            LOGE("NGX AllocateParameters failed 0x%08X", r);
            return false;
        }
    }
    NVSDK_NGX_Parameter* p = v.params;
    p->vt->Reset(p);
    p->SetUI("CreationNodeMask", 1);
    p->SetUI("VisibilityNodeMask", 1);
    p->SetUI("Width", v.lowW);
    p->SetUI("Height", v.lowH);
    p->SetUI("OutWidth", v.fullW);
    p->SetUI("OutHeight", v.fullH);
    p->SetI("PerfQualityValue", v.quality);
    p->SetI("RTXValue", 0);
    p->SetI("DLSS.Feature.Create.Flags", v.flags);
    p->SetI("DLSS.Enable.Output.Subrects", 0);
    const char* presetKeys[] = {"DLSS.Hint.Render.Preset.DLAA",        "DLSS.Hint.Render.Preset.Quality",
                                "DLSS.Hint.Render.Preset.Balanced",    "DLSS.Hint.Render.Preset.Performance",
                                "DLSS.Hint.Render.Preset.UltraPerformance", "DLSS.Hint.Render.Preset.UltraQuality"};
    for (const char* k : presetKeys)
        p->SetUI(k, (unsigned)v.preset);

    NVSDK_NGX_Result r = g_ngx.CreateFeature(imm, NVSDK_NGX_Feature_SuperSampling, p, &v.feature);
    LOGI("DLSS CreateFeature %ux%u -> %ux%u quality=%d flags=0x%X preset=%d -> 0x%08X handle=%p", v.lowW, v.lowH, v.fullW,
         v.fullH, v.quality, v.flags, v.preset, r, v.feature);
    if (!NGX_OK(r) || !v.feature)
    {
        v.feature = nullptr;
        return false;
    }
    v.reset = true;
    return true;
}

static void LogTexture(const char* what, ID3D11View* view)
{
    if (!view)
    {
        LOGI("  %-12s: (null)", what);
        return;
    }
    ID3D11Resource* r = nullptr;
    view->GetResource(&r);
    D3D11_RESOURCE_DIMENSION dim;
    r->GetType(&dim);
    if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D)
    {
        D3D11_TEXTURE2D_DESC d;
        ((ID3D11Texture2D*)r)->GetDesc(&d);
        LOGI("  %-12s: view %p tex %p %ux%u fmt=%d mips=%u arr=%u samples=%u bind=0x%X misc=0x%X", what, view, r, d.Width,
             d.Height, d.Format, d.MipLevels, d.ArraySize, d.SampleDesc.Count, d.BindFlags, d.MiscFlags);
    }
    else
        LOGI("  %-12s: view %p resource %p dim=%d", what, view, r, dim);
    r->Release();
}

static volatile LONG s_mvStatsFrames;
static void PollCameraInfo(ID3D11DeviceContext* imm)
{
    if (!s_infoPending)
        return;
    for (int i = 0; i < 3; i++)
    {
        if (!(s_infoPending & (1 << i)) || !s_infoStaging[i])
            continue;
        D3D11_MAPPED_SUBRESOURCE m;
        HRESULT hr = imm->Map(s_infoStaging[i], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &m);
        if (hr == S_OK)
        {
            const float* f = (const float*)m.pData;
            if (s_mvStatsFrames > 0 && f[5] == 1.0f)
            {
                s_mvStatsFrames--;
                LOGI("MVStats obj n=%.0f game(%.2f,%.2f) reproj(%.2f,%.2f) |game-reproj|=%.2f  static n=%.0f mv(%.2f,%.2f) | "
                     "charbox n=%.0f err used=%.4f reproj=%.4f zero=%.4f |mv| used=%.1f reproj=%.1f | rest n=%.0f err reproj=%.4f "
                     "zero=%.4f",
                     f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15], f[16], f[19], f[20], f[21], f[22], f[23], f[24],
                     f[25], f[26], f[27]);
            }
            if (f[5] == 1.0f)
            {
                float rev = f[0];
                if (!s_camValid || rev != s_camRev)
                    LOGI("Camera info: reversedZ=%d near=%.4f far=%.1f P11=%.4f P00=%.4f (vFOV %.2f deg)", (int)rev, f[1],
                         f[2], f[3], f[4], 2.0 * atan(1.0 / (fabs(f[3]) > 1e-6 ? fabs(f[3]) : 1.0)) * 57.29578);
                s_camRev = rev;
                s_camNear = f[1];
                float farZ = f[2];
                if (!(farZ > 0.0f && farZ < 1.0e7f)) // infinite far plane (reversed-Z): use a large finite value
                    farZ = 100000.0f;
                s_camFar = farZ;
                s_camP11 = f[3];
                s_camP00 = f[4];
                s_camValid = true;
            }
            imm->Unmap(s_infoStaging[i], 0);
            s_infoPending &= ~(1 << i);
        }
    }
}


// ------------------------------------------------------------------ compute-stage state save/restore
// (SwapDeviceContextState can't be used: on this D3D11 layer it rewrites the context's vtable.)
struct CsState
{
    ID3D11ComputeShader* cs;
    ID3D11ClassInstance* ci[256];
    UINT nci;
    ID3D11Buffer* cb[14];
    ID3D11ShaderResourceView* srv[32];
    ID3D11UnorderedAccessView* uav[8];
    ID3D11SamplerState* smp[16];
};
static CsState s_csSave;

static void SaveCs(ID3D11DeviceContext* c, CsState& st)
{
    st.nci = 256;
    c->CSGetShader(&st.cs, st.ci, &st.nci);
    c->CSGetConstantBuffers(0, 14, st.cb);
    c->CSGetShaderResources(0, 32, st.srv);
    c->CSGetUnorderedAccessViews(0, 8, st.uav);
    c->CSGetSamplers(0, 16, st.smp);
}

static void RestoreCs(ID3D11DeviceContext* c, CsState& st)
{
    c->CSSetShader(st.cs, st.nci ? st.ci : nullptr, st.nci);
    c->CSSetConstantBuffers(0, 14, st.cb);
    c->CSSetShaderResources(0, 32, st.srv);
    UINT counts[8];
    for (int i = 0; i < 8; i++)
        counts[i] = (UINT)-1;
    c->CSSetUnorderedAccessViews(0, 8, st.uav, counts);
    c->CSSetSamplers(0, 16, st.smp);
    SafeRelease(st.cs);
    for (UINT i = 0; i < st.nci; i++)
        SafeRelease(st.ci[i]);
    for (auto*& x : st.cb)
        SafeRelease(x);
    for (auto*& x : st.srv)
        SafeRelease(x);
    for (auto*& x : st.uav)
        SafeRelease(x);
    for (auto*& x : st.smp)
        SafeRelease(x);
}

static ID3D11DeviceContext* volatile s_immCtx; // the game's immediate context (seen in ExecuteCommandList)

// ------------------------------------------------------------------ DLSS execution (immediate context)
static void NoteEval(LONG64 taaIndex);
void SlCamUpdate(const uint8_t* pass, const float* reproj, const float* jitter, const float* low);
void SlCamInit();
void SlCamStats(char* out, size_t n);
static volatile LONG s_evalTrace;
static uint64_t s_traceT0;
static bool RunJob(ID3D11DeviceContext* imm, Job* job)
{
    if (s_fatal)
        return false;
    ID3D11DeviceContext1* imm1 = nullptr;
    imm->QueryInterface(__uuidof(ID3D11DeviceContext1), (void**)&imm1);
    if (!imm1)
        return false;

    if (!s_ngxInitDone)
    {
        s_ngxInitDone = true;
        if (!NgxInit(s_dev) || !NgxDlssAvailable(s_dev))
        {
            s_fatal = true;
            LOGE("NGX/DLSS unavailable - mod disabled, game TAA stays active from now on");
            imm1->Release();
            return false;
        }
    }
    PollCameraInfo(imm);

    AcquireSRWLockExclusive(&s_viewLock);
    View* v = job->view;
    bool ok = false;
    if (v->gen != job->gen || !v->texOut)
    {
        LOGD("Job skipped: view resources changed");
        goto out;
    }
    SaveCs(imm, s_csSave);

    if (!v->feature && !CreateFeature(imm, *v))
    {
        if (InterlockedIncrement64(&s_jobFails) >= 3)
        {
            s_fatal = true;
            LOGE("DLSS feature creation keeps failing - mod disabled");
        }
        goto swapback;
    }

    {
        uint32_t modFlags[4] = {(job->hdr ? 1u : 0u) | (g_cfg.useBiasMask ? 2u : 0u) | (g_cfg.debugView == 5 ? 4u : 0u), 0, 0, 0};
        imm->UpdateSubresource(s_modCB_CS, 0, nullptr, modFlags, 0, 0);
        imm->UpdateSubresource(s_taaCB, 0, nullptr, &job->tc, 0, 0);
        imm->UpdateSubresource(s_passCB, 0, nullptr, job->pass, 0, 0);
        ID3D11Buffer* cbs[6] = {s_modCB_CS, nullptr, s_passCB, nullptr, nullptr, s_taaCB};
        imm->CSSetConstantBuffers(0, 6, cbs);
        ID3D11ShaderResourceView* ins[5] = {job->color, job->motion, job->depth, job->resp ? job->resp : s_dummySRV,
                                            v->srvPrev ? v->srvPrev : s_dummySRV};
        imm->CSSetShaderResources(0, 5, ins);
        ID3D11UnorderedAccessView* outs[5] = {v->uavColor, v->uavMV, v->uavDepth, v->uavBias, s_infoUAV};
        imm->CSSetUnorderedAccessViews(0, 5, outs, nullptr);
        imm->CSSetShader(s_prepCS, nullptr, 0);
        imm->Dispatch((v->lowW + 7) / 8, (v->lowH + 7) / 8, 1);
        ID3D11UnorderedAccessView* nullU[5] = {};
        ID3D11ShaderResourceView* nullS[5] = {};
        ID3D11Buffer* nullB[6] = {};
        imm->CSSetUnorderedAccessViews(0, 5, nullU, nullptr);
        imm->CSSetShaderResources(0, 5, nullS);
        if (v->texPrev && (s_mvStatsFrames > 0 || g_cfg.debugView == 5))
            imm->CopyResource(v->texPrev, v->texColor);
        if (v->uavMVHi && s_dilateCS)
        {
            uint32_t dflags[4] = {(v->flags & NGX_DLSS_DepthInverted) ? 1u : 0u, 0, 0, 0};
            imm->UpdateSubresource(s_modCB_CS, 0, nullptr, dflags, 0, 0);
            ID3D11ShaderResourceView* din[2] = {v->srvMV, v->srvDepth};
            imm->CSSetShaderResources(0, 2, din);
            imm->CSSetUnorderedAccessViews(0, 1, &v->uavMVHi, nullptr);
            imm->CSSetShader(s_dilateCS, nullptr, 0);
            imm->Dispatch((v->fullW + 7) / 8, (v->fullH + 7) / 8, 1);
            imm->CSSetUnorderedAccessViews(0, 5, nullU, nullptr);
            imm->CSSetShaderResources(0, 5, nullS);
        }
        imm->CSSetConstantBuffers(0, 6, nullB);
        imm->CSSetShader(nullptr, nullptr, 0);

        if (((job->taaIndex % 30) == 1 || s_mvStatsFrames > 0) && s_infoBuf)
        {
            int i = s_infoWrite;
            if (!(s_infoPending & (1 << i)) && s_infoStaging[i])
            {
                imm->CopyResource(s_infoStaging[i], s_infoBuf);
                s_infoPending |= 1 << i;
                s_infoWrite = (i + 1) % 3;
            }
        }
    }

    {
        uint64_t now = NowQpc();
        float dtMs = v->lastQpc ? (float)QpcToMs(now - v->lastQpc) : 16.6f;
        if (dtMs > 1000.0f)
            dtMs = 1000.0f;
        v->lastQpc = now;
        const TaaConstants& tc = job->tc;
        float jx = tc.subPixelJitter[0] * g_cfg.jitterSignX, jy = tc.subPixelJitter[1] * g_cfg.jitterSignY;
        bool reset = v->reset || job->reset;

        NVSDK_NGX_Parameter* p = v->params;
        p->SetRes("Color", v->texColor);
        p->SetRes("Output", v->texOut);
        p->SetRes("Depth", v->texDepth);
        bool hiMV = v->texMVHi != nullptr;
        p->SetRes("MotionVectors", hiMV ? v->texMVHi : v->texMV);
        p->SetRes("DLSS.Input.Bias.Current.Color.Mask", (g_cfg.useBiasMask && g_cfg.debugView != 5) ? v->texBias : nullptr);
        p->SetRes("ExposureTexture", nullptr);
        p->SetF("Jitter.Offset.X", jx);
        p->SetF("Jitter.Offset.Y", jy);
        p->SetF("MV.Scale.X", (float)(hiMV ? v->fullW : v->lowW));
        p->SetF("MV.Scale.Y", (float)(hiMV ? v->fullH : v->lowH));
        p->SetI("Reset", reset ? 1 : 0);
        p->SetF("Sharpness", g_cfg.sharpness);
        p->SetUI("DLSS.Render.Subrect.Dimensions.Width", v->lowW);
        p->SetUI("DLSS.Render.Subrect.Dimensions.Height", v->lowH);
        p->SetUI("DLSS.Input.Color.Subrect.Base.X", 0);
        p->SetUI("DLSS.Input.Color.Subrect.Base.Y", 0);
        p->SetUI("DLSS.Input.Depth.Subrect.Base.X", 0);
        p->SetUI("DLSS.Input.Depth.Subrect.Base.Y", 0);
        p->SetUI("DLSS.Input.MV.Subrect.Base.X", 0);
        p->SetUI("DLSS.Input.MV.Subrect.Base.Y", 0);
        p->SetUI("DLSS.Output.Subrect.Base.X", 0);
        p->SetUI("DLSS.Output.Subrect.Base.Y", 0);
        p->SetF("DLSS.Pre.Exposure", 1.0f);
        p->SetF("DLSS.Exposure.Scale", 1.0f);
        p->SetF("FrameTimeDeltaInMsec", dtMs);
        // camera data for OptiScaler's frame generation (ignored by DLSS itself)
        float vfov = 2.0f * (float)atan(1.0 / (fabs(s_camP11) > 1e-6f ? fabs(s_camP11) : 1.0));
        p->SetF("FSR.cameraNear", s_camNear > 0.0f ? s_camNear : 0.1f);
        p->SetF("FSR.cameraFar", s_camFar > 0.0f ? s_camFar : 10000.0f);
        p->SetF("FSR.cameraFovAngleVertical", vfov);
        p->SetF("FSR.frameTimeDelta", dtMs);
        p->SetF("FSR.viewSpaceToMetersFactor", 1.0f);

        NVSDK_NGX_Result r = g_ngx.EvaluateFeature(imm, v->feature, p, nullptr);
        v->evalCount++;
        NoteEval(job->taaIndex);
        if (g_cfg.fgCameraData)
            SlCamUpdate(job->pass, job->tc.reproj, job->tc.subPixelJitter, job->tc.lowResSize);
        if (!NGX_OK(r))
        {
            LOGE("DLSS EvaluateFeature failed 0x%08X", r);
            if (InterlockedIncrement64(&s_jobFails) >= 10)
            {
                s_fatal = true;
                LOGE("Too many DLSS evaluation failures - mod disabled");
            }
            goto swapback;
        }
        if (v->evalCount <= 16)
            LOGD("  eval %I64u: jitter (%.4f, %.4f) reset %d dt %.2fms", v->evalCount, jx, jy, reset ? 1 : 0, dtMs);
        if (v->evalCount == 1)
            LOGI("DLSS evaluate OK (first frame) %ux%u -> %ux%u", v->lowW, v->lowH, v->fullW, v->fullH);
        v->reset = false;
        ok = true;
    }

swapback:
    RestoreCs(imm, s_csSave);
out:
    ReleaseSRWLockExclusive(&s_viewLock);
    imm1->Release();
    if (ok)
        InterlockedIncrement64(&s_dlssFrames);
        s_lastEvalQpc = NowQpc();
    return ok;
}


// ------------------------------------------------------------------ pipeline trace (diagnostics)
// Records, for one frame, every draw/dispatch that consumes the TAA output (directly or through
// intermediate targets), with shader names taken from the shaders' reflection data.
struct ShaderName
{
    void* sh;
    char name[64];
};
static ShaderName s_shNames[6144];
static volatile LONG s_shNameCount;

static void NameFromBytecode(const void* code, SIZE_T len, char* out, size_t outLen)
{
    out[0] = 0;
    const uint8_t* b = (const uint8_t*)code;
    if (len < 32 || memcmp(b, "DXBC", 4) != 0)
        return;
    uint32_t nch = *(const uint32_t*)(b + 28);
    for (uint32_t i = 0; i < nch && 32 + i * 4 + 4 <= len; i++)
    {
        uint32_t off = *(const uint32_t*)(b + 32 + i * 4);
        if (off + 8 > len || memcmp(b + off, "RDEF", 4) != 0)
            continue;
        const uint8_t* r = b + off + 8;
        uint32_t rlen = *(const uint32_t*)(b + off + 4);
        uint32_t bc = *(const uint32_t*)(r + 8), bo = *(const uint32_t*)(r + 12);
        size_t n = 0;
        for (uint32_t k = 0; k < bc && bo + k * 32 + 32 <= rlen; k++)
        {
            const uint8_t* e = r + bo + k * 32;
            uint32_t no = *(const uint32_t*)e, type = *(const uint32_t*)(e + 4);
            if (type == 3 || no >= rlen) // samplers
                continue;
            const char* nm = (const char*)(r + no);
            size_t l = strlen(nm);
            if (n + l + 2 >= outLen)
                break;
            if (n)
                out[n++] = ',';
            memcpy(out + n, nm, l);
            n += l;
            out[n] = 0;
        }
        return;
    }
}

static void RegisterShaderName(void* sh, const void* code, SIZE_T len)
{
    LONG i = InterlockedIncrement(&s_shNameCount) - 1;
    if (i >= 6144)
        return;
    s_shNames[i].sh = sh;
    NameFromBytecode(code, len, s_shNames[i].name, sizeof(s_shNames[i].name));
}

static const char* ShaderNameOf(void* sh)
{
    LONG n = s_shNameCount < 6144 ? s_shNameCount : 6144;
    for (LONG i = n - 1; i >= 0; i--)
        if (s_shNames[i].sh == sh)
            return s_shNames[i].name;
    return "?";
}

static volatile LONG s_traceActive;
static uint64_t s_traceEndQpc;
static ID3D11Resource* s_traceSet[64];
static int s_traceSetCount;
static SRWLOCK s_traceLock = SRWLOCK_INIT;
static int s_traceLogged;

static bool TraceHas(ID3D11Resource* r)
{
    for (int i = 0; i < s_traceSetCount; i++)
        if (s_traceSet[i] == r)
            return true;
    return false;
}

static void TraceStart(ID3D11Resource* taaOut)
{
    AcquireSRWLockExclusive(&s_traceLock);
    s_traceSetCount = 0;
    s_traceSet[s_traceSetCount++] = taaOut;
    s_traceLogged = 0;
    ReleaseSRWLockExclusive(&s_traceLock);
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    s_traceEndQpc = NowQpc() + (uint64_t)f.QuadPart / 10; // 100 ms
    s_traceActive = 1;
    LOGI("TRACE start: consumers of TAA output %p", taaOut);
}

static void TraceDescribe(ID3D11View* v, char* out, size_t n)
{
    out[0] = 0;
    if (!v)
        return;
    ID3D11Resource* r = nullptr;
    v->GetResource(&r);
    D3D11_RESOURCE_DIMENSION dim;
    r->GetType(&dim);
    if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D)
    {
        D3D11_TEXTURE2D_DESC d;
        ((ID3D11Texture2D*)r)->GetDesc(&d);
        _snprintf(out, n, "%p %ux%u fmt%d", r, d.Width, d.Height, d.Format);
    }
    else
        _snprintf(out, n, "%p dim%d", r, dim);
    r->Release();
}

// returns true if the draw reads a traced resource
static void TraceDraw(ID3D11DeviceContext* ctx, bool compute)
{
    if (!s_traceActive)
        return;
    if (NowQpc() > s_traceEndQpc)
    {
        s_traceActive = 0;
        LOGI("TRACE end");
        return;
    }
    ID3D11ShaderResourceView* srv[16] = {};
    if (compute)
        ctx->CSGetShaderResources(0, 16, srv);
    else
        ctx->PSGetShaderResources(0, 16, srv);
    int hit = -1;
    AcquireSRWLockExclusive(&s_traceLock);
    for (int i = 0; i < 16 && hit < 0; i++)
        if (srv[i])
        {
            ID3D11Resource* r = nullptr;
            srv[i]->GetResource(&r);
            if (TraceHas(r))
                hit = i;
            r->Release();
        }
    if (hit >= 0 && s_traceLogged < 40)
    {
        s_traceLogged++;
        void* sh = nullptr;
        ID3D11View* outv = nullptr;
        if (compute)
        {
            ID3D11ComputeShader* cs = nullptr;
            ctx->CSGetShader(&cs, nullptr, nullptr);
            sh = cs;
            if (cs)
                cs->Release();
            ID3D11UnorderedAccessView* u = nullptr;
            ctx->CSGetUnorderedAccessViews(0, 1, &u);
            outv = u;
        }
        else
        {
            ID3D11PixelShader* ps = nullptr;
            ctx->PSGetShader(&ps, nullptr, nullptr);
            sh = ps;
            if (ps)
                ps->Release();
            ID3D11RenderTargetView* rt = nullptr;
            ctx->OMGetRenderTargets(1, &rt, nullptr);
            outv = rt;
        }
        char in[96], o[96];
        TraceDescribe(srv[hit], in, sizeof(in));
        TraceDescribe(outv, o, sizeof(o));
        D3D11_VIEWPORT vp = {};
        UINT nvp = 1;
        ctx->RSGetViewports(&nvp, &vp);
        LOGI("TRACE %s ctx %p shader %p [%s] reads t%d (%s) -> %s vp %.0fx%.0f", compute ? "CS" : "PS", ctx, sh,
             ShaderNameOf(sh), hit, in, o, vp.Width, vp.Height);
        if (outv)
        {
            ID3D11Resource* r = nullptr;
            outv->GetResource(&r);
            if (!TraceHas(r) && s_traceSetCount < 64)
                s_traceSet[s_traceSetCount++] = r;
            r->Release();
            outv->Release();
        }
    }
    ReleaseSRWLockExclusive(&s_traceLock);
    for (auto* x : srv)
        SafeRelease(x);
}

// ------------------------------------------------------------------ the TAA replacement (recording side)
struct DrawArgs
{
    int kind; // 0 Draw, 1 DrawIndexed, 2 DrawInstanced, 3 DrawIndexedInstanced
    UINT a, b, c, d;
    INT e;
};

static void CallOriginalDraw(ID3D11DeviceContext* ctx, VtHooks* h, const DrawArgs& d)
{
    switch (d.kind)
    {
    case 0: ORIG(h, kCtx_Draw, PFN_Draw)(ctx, d.a, d.b); break;
    case 1: ORIG(h, kCtx_DrawIndexed, PFN_DrawIndexed)(ctx, d.a, d.b, d.e); break;
    case 2: ORIG(h, kCtx_DrawInstanced, PFN_DrawInstanced)(ctx, d.a, d.b, d.c, d.d); break;
    case 3: ORIG(h, kCtx_DrawIndexedInstanced, PFN_DrawIndexedInstanced)(ctx, d.a, d.b, d.c, d.e, d.d); break;
    }
}

static volatile LONG s_detailLogs = 2;
static uint64_t s_lastStatQpc;
static volatile LONG64 s_noShadowFrames;

static void RecordResolveDraw(ID3D11DeviceContext* ctx, VtHooks* h, ID3D11PixelShader* taaPS, View* v,
                              const DrawArgs& da)
{
    ID3D11ShaderResourceView* savedSrv[3] = {};
    ID3D11Buffer* savedCb = nullptr;
    ctx->PSGetShaderResources(0, 3, savedSrv);
    ctx->PSGetConstantBuffers(0, 1, &savedCb);

    uint32_t psFlags[4] = {(uint32_t)g_cfg.debugView, 0, 0, 0};
    ctx->UpdateSubresource(s_modCB_PS, 0, nullptr, psFlags, 0, 0);
    ID3D11ShaderResourceView* mine[3] = {v->srvOut, v->srvMV,
                                         (g_cfg.debugView == 5 && v->srvBias) ? v->srvBias : v->srvDepth};
    if (g_cfg.debugView == 6 && s_shadowMaskSrv)
        mine[2] = s_shadowMaskSrv;
    ctx->PSSetShaderResources(0, 3, mine);
    ctx->PSSetConstantBuffers(0, 1, &s_modCB_PS);
    ORIG(h, kCtx_PSSetShader, PFN_PSSetShader)(ctx, s_resolvePS, nullptr, 0);

    CallOriginalDraw(ctx, h, da);

    ORIG(h, kCtx_PSSetShader, PFN_PSSetShader)(ctx, taaPS, nullptr, 0);
    ctx->PSSetConstantBuffers(0, 1, &savedCb);
    ctx->PSSetShaderResources(0, 3, savedSrv);
    for (auto* s : savedSrv)
        SafeRelease(s);
    SafeRelease(savedCb);
}


// ------------------------------------------------------------------ live tuning (developer aid)
// When [Log] DevCommands=1, GRB_DLSS_cmd.ini is re-read whenever it changes, so settings can be
// switched without keyboard input.
void RenderScaleApply(float scale);
float RenderScaleGameRatio();
float RenderScaleCurrent();
static float s_renderScale = -1.0f; // runtime value (hotkeys); -1 = take from config
static float s_cfgRenderScale = -1.0f;
void OverlayToast(const char* text);
void OverlayInit();
void OverlayHookDeviceFactory(ID3D11Device* dev);

float ModGetRenderScale()
{
    return s_renderScale > 0.0f ? s_renderScale : 0.0f;
}
void ModSetRenderScale(float v)
{
    if (s_cfgRenderScale < -1.5f || s_cfgRenderScale != g_cfg.renderScale)
        s_cfgRenderScale = g_cfg.renderScale; // don't let the config re-apply over the menu choice
    s_renderScale = v > 0.0f ? v : 0.0f;
    RenderScaleApply(s_renderScale);
}
bool ModGetDlssOn()
{
    return s_userEnabled;
}
void ModSetDlssOn(bool on)
{
    s_userEnabled = on;
    for (View& v : s_views)
        v.reset = true;
    LOGI("Menu: DLSS %s", on ? "ON" : "OFF (game TAA)");
}
void ModGetStatus(UINT* lowW, UINT* lowH, UINT* fullW, UINT* fullH, bool* active, bool* seen)
{
    *lowW = s_stLowW;
    *lowH = s_stLowH;
    *fullW = s_stFullW;
    *fullH = s_stFullH;
    uint64_t now = NowQpc();
    *active = s_lastEvalQpc && QpcToMs(now - s_lastEvalQpc) < 500.0;
    *seen = s_lastTaaQpc && QpcToMs(now - s_lastTaaQpc) < 500.0;
}
void ModSetOverlayThread(bool on)
{
    s_modThread = on ? GetCurrentThreadId() : 0;
}
void ProbeOnTaa(const float* low, const float* full, const float* frame16, bool haveFrame);
void ProbePoll(const wchar_t* ini);
static FILETIME s_cmdTime, s_iniTime;
static void PollCommands()
{
    // GRB_DLSS.ini is re-read whenever it is saved: presets/quality apply without restarting the game
    {
        wchar_t ip[MAX_PATH];
        lstrcpyW(ip, g_modDir);
        lstrcatW(ip, L"GRB_DLSS.ini");
        WIN32_FILE_ATTRIBUTE_DATA ia;
        if (GetFileAttributesExW(ip, GetFileExInfoStandard, &ia))
        {
            if (s_iniTime.dwLowDateTime == 0 && s_iniTime.dwHighDateTime == 0)
                s_iniTime = ia.ftLastWriteTime;
            else if (CompareFileTime(&ia.ftLastWriteTime, &s_iniTime) != 0)
            {
                s_iniTime = ia.ftLastWriteTime;
                LoadConfig();
                LOGI("GRB_DLSS.ini reloaded: Enabled=%d Preset=%d QualityMode=%d RenderScale=%.3f", g_cfg.enabled,
                     g_cfg.preset, g_cfg.qualityOverride, g_cfg.renderScale);
                s_cfgRenderScale = -2.0f; // re-apply the file's RenderScale
            }
        }
    }
    if (!g_cfg.devCommands)
        return;
    wchar_t p[MAX_PATH];
    lstrcpyW(p, g_modDir);
    lstrcatW(p, L"GRB_DLSS_cmd.ini");
    WIN32_FILE_ATTRIBUTE_DATA fa;
    if (!GetFileAttributesExW(p, GetFileExInfoStandard, &fa))
        return;
    if (CompareFileTime(&fa.ftLastWriteTime, &s_cmdTime) == 0)
        return;
    s_cmdTime = fa.ftLastWriteTime;
    wchar_t b[64];
    GetPrivateProfileStringW(L"Cmd", L"DLSS", L"", b, 64, p);
    if (b[0])
        s_userEnabled = b[0] == L'1';
    GetPrivateProfileStringW(L"Cmd", L"JitterScaleX", L"", b, 64, p);
    if (b[0])
        g_cfg.jitterSignX = (b[0] == L'-') ? -1.0f : (b[0] == L'0' ? 0.0f : 1.0f);
    GetPrivateProfileStringW(L"Cmd", L"JitterScaleY", L"", b, 64, p);
    if (b[0])
        g_cfg.jitterSignY = (b[0] == L'-') ? -1.0f : (b[0] == L'0' ? 0.0f : 1.0f);
    g_cfg.debugView = GetPrivateProfileIntW(L"Cmd", L"DebugView", g_cfg.debugView, p);
    g_cfg.useBiasMask = GetPrivateProfileIntW(L"Cmd", L"UseResponsiveMask", g_cfg.useBiasMask, p);
    {
        static int s_etSeq = -1;
        int seq = GetPrivateProfileIntW(L"Cmd", L"EvalTrace", -1, p);
        if (seq != s_etSeq)
        {
            if (s_etSeq != -1 || seq > 0)
            {
                s_traceT0 = NowQpc();
                s_evalTrace = 200;
            }
            s_etSeq = seq;
        }
    }
    {
        static int s_mvSeq = -1;
        int seq = GetPrivateProfileIntW(L"Cmd", L"MVStats", -1, p);
        if (seq != s_mvSeq)
        {
            if (s_mvSeq != -1 || seq > 0)
                s_mvStatsFrames = GetPrivateProfileIntW(L"Cmd", L"MVStatsFrames", 300, p);
            s_mvSeq = seq;
        }
    }
    for (View& v : s_views)
        v.reset = true;
    LOGI("Commands: DLSS %s, jitter scale (%.0f, %.0f), debug view %d, responsive mask %d", s_userEnabled ? "on" : "off",
         g_cfg.jitterSignX, g_cfg.jitterSignY, g_cfg.debugView, g_cfg.useBiasMask);
    ProbePoll(p);
}

static bool HandleTaa(ID3D11DeviceContext* ctx, VtHooks* h, CtxInfo* ce, ID3D11PixelShader* taaPS, const DrawArgs& da)
{
    LONG64 taaIndex = InterlockedIncrement64(&s_taaFrames);
    if ((taaIndex % 30) == 0)
        PollCommands();

    if (g_cfg.toggleKey && (GetAsyncKeyState(g_cfg.toggleKey) & 1))
    {
        s_userEnabled = !s_userEnabled;
        LOGI("Toggle: DLSS %s", s_userEnabled ? "ON" : "OFF (game TAA)");
        for (View& v : s_views)
            v.reset = true;
    }
    if (g_cfg.presetKey && (GetAsyncKeyState(g_cfg.presetKey) & 1))
    {
        static const int cycle[] = {0, 10, 11, 12, 13}; // default, J, K, L, M
        int next = 0;
        for (int i = 0; i < 5; i++)
            if (cycle[i] == g_cfg.preset)
                next = (i + 1) % 5;
        g_cfg.preset = cycle[next];
        LOGI("Preset key: DLSS preset now %s", g_cfg.preset ? (const char[][2]){"J", "K", "L", "M"}[g_cfg.preset - 10] : "default");
    }
    // custom internal render resolution
    if (g_cfg.renderScale != s_cfgRenderScale) // startup or GRB_DLSS.ini edited
    {
        s_cfgRenderScale = g_cfg.renderScale;
        s_renderScale = g_cfg.renderScale;
    }
    {
        int dir = 0;
        if (g_cfg.renderScaleUpKey && (GetAsyncKeyState(g_cfg.renderScaleUpKey) & 1))
            dir = 1;
        if (g_cfg.renderScaleDownKey && (GetAsyncKeyState(g_cfg.renderScaleDownKey) & 1))
            dir = -1;
        if (dir)
        {
            float cur = s_renderScale > 0.0f ? s_renderScale : RenderScaleGameRatio();
            if (cur <= 0.0f)
                cur = 0.667f;
            float step = g_cfg.renderScaleStep > 0.0f ? g_cfg.renderScaleStep : 0.05f;
            // snap to the step grid
            float n = (float)floor(cur / step + 0.5);
            float next = (n + dir) * step;
            if (next > 1.0f)
                next = 1.0f;
            if (next < 0.25f)
                next = 0.25f;
            s_renderScale = next;
            LOGI("RenderScale key: %.0f%%", next * 100.0f);
            char t[96];
            if (s_stFullW)
                _snprintf(t, sizeof(t), "Render resolution %d%%  (%ux%u)", (int)(next * 100.0f + 0.5f),
                          (UINT)(s_stFullW * next), (UINT)(s_stFullH * next));
            else
                _snprintf(t, sizeof(t), "Render resolution %d%%", (int)(next * 100.0f + 0.5f));
            OverlayToast(t);
        }
    }
    if (g_cfg.enabled && (s_renderScale > 0.0f || RenderScaleCurrent() > 0.0f || g_cfg.renderScaleUpKey ||
                          g_cfg.renderScaleDownKey))
        RenderScaleApply(s_renderScale > 0.0f ? s_renderScale : 0.0f);

    if (!g_cfg.enabled || s_fatal || !s_userEnabled)
        return false;

    if (ce->type < 0)
        ce->type = ctx->GetType() == D3D11_DEVICE_CONTEXT_IMMEDIATE ? 0 : 1;
    const bool deferred = ce->type != 0;
    if (deferred && ce->pendingCount >= 4)
        return false;

    ID3D11ShaderResourceView* srv[11] = {};
    ctx->PSGetShaderResources(13, 11, srv);
    ID3D11RenderTargetView* rtv[3] = {};
    ID3D11DepthStencilView* dsv = nullptr;
    ctx->OMGetRenderTargets(3, rtv, &dsv);
    ID3D11Buffer* cb[6] = {};
    UINT cbFirst[6] = {}, cbNum[6] = {};
    ID3D11DeviceContext1* ctx1 = nullptr;
    ctx->QueryInterface(__uuidof(ID3D11DeviceContext1), (void**)&ctx1);
    if (ctx1)
        ctx1->PSGetConstantBuffers1(0, 6, cb, cbFirst, cbNum);
    else
        ctx->PSGetConstantBuffers(0, 6, cb);

    ID3D11ShaderResourceView* srvColor = srv[0];
    ID3D11ShaderResourceView* srvMotion = srv[1];
    ID3D11ShaderResourceView* srvDepth = srv[2];
    ID3D11ShaderResourceView* srvResp = srv[3];

    bool ok = false;
    bool detail = InterlockedDecrement(&s_detailLogs) >= 0;
    Job* job = nullptr;
    View* v = nullptr;

    if (detail)
    {
        LOGI("TAA draw #%I64u (ctx %p %s, ps %p, draw kind %d args %u %u):", taaIndex, ctx,
             deferred ? "deferred" : "immediate", taaPS, da.kind, da.a, da.b);
        LogTexture("t13 color", srvColor);
        LogTexture("t14 motion", srvMotion);
        LogTexture("t15 depth", srvDepth);
        LogTexture("t16 respons.", srvResp);
        LogTexture("t18 history", srv[5]);
        LogTexture("RT0 output", rtv[0]);
        LogTexture("RT1 history", rtv[1]);
        LogTexture("RT2 hist.a", rtv[2]);
        for (int i = 0; i < 6; i++)
            if (cb[i])
            {
                D3D11_BUFFER_DESC bd;
                cb[i]->GetDesc(&bd);
                LOGI("  cb%d: %p ByteWidth=%u Usage=%u CPU=0x%X first=%u num=%u", i, cb[i], bd.ByteWidth, bd.Usage,
                     bd.CPUAccessFlags, cbFirst[i], cbNum[i]);
            }
        D3D11_VIEWPORT vp;
        UINT nvp = 1;
        ctx->RSGetViewports(&nvp, &vp);
        LOGI("  viewport %.0fx%.0f @ %.0f,%.0f", vp.Width, vp.Height, vp.TopLeftX, vp.TopLeftY);
    }

    if (!srvColor || !srvMotion || !srvDepth || !rtv[0] || !cb[5] || !cb[2])
    {
        if (detail)
            LOGE("TAA pass bindings incomplete (color %p motion %p depth %p rt0 %p cb5 %p cb2 %p)", srvColor, srvMotion,
                 srvDepth, rtv[0], cb[5], cb[2]);
        goto done;
    }

    WatchBuffer(cb[5]);
    WatchBuffer(cb[2]);

    job = (Job*)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(Job));
    if (!job)
        goto done;
    {
        bool same5 = false, same2 = false;
        bool have5 = ReadShadow(cb[5], ctx, cbFirst[5] * 16, &job->tc, sizeof(TaaConstants), &same5);
        static uint8_t s_lastPass[kPassBytes];
        static volatile bool s_havePass;
        bool have2 = ReadShadow(cb[2], ctx, cbFirst[2] * 16, job->pass, kPassBytes, &same2);
        if (have2)
        {
            memcpy(s_lastPass, job->pass, kPassBytes);
            s_havePass = true;
        }
        else if (s_havePass)
        {
            // the projection data is (nearly) constant: reuse the last capture for a not-yet-seen buffer
            memcpy(job->pass, s_lastPass, kPassBytes);
            have2 = true;
        }
        if (!have5 || !have2)
        {
            LONG64 n = InterlockedIncrement64(&s_noShadowFrames);
            if (n == 120)
                LOGE("TAA constants still not captured after 120 TAA passes (taa %d, pass %d, cb updates seen %I64u, gpu "
                     "copies %I64u) - using game TAA",
                     have5, have2, s_cbUpdates, s_cbGpuCopies);
            goto done;
        }
        if (detail)
            LOGI("  constants captured from %s context(s)", same5 && same2 ? "the same" : "another");
    }

    {
        const TaaConstants& tc = job->tc;
        UINT lowW = (UINT)(tc.lowResSize[0] + 0.5f), lowH = (UINT)(tc.lowResSize[1] + 0.5f);
        UINT fullW = (UINT)(tc.fullResSize[0] + 0.5f), fullH = (UINT)(tc.fullResSize[1] + 0.5f);
        s_stLowW = lowW;
        s_stLowH = lowH;
        s_stFullW = fullW;
        s_stFullH = fullH;
        s_lastTaaQpc = NowQpc();
        if (g_cfg.devCommands)
        {
            float fr[8] = {};
            bool haveF = false;
            if (cb[1])
            {
                WatchBuffer(cb[1]);
                haveF = ReadShadow(cb[1], ctx, cbFirst[1] * 16 + 1440, fr, 32, nullptr);
            }
            ProbeOnTaa(tc.lowResSize, tc.fullResSize, fr, haveF);
        }
        if (g_cfg.shadowDiag)
        {
            // jitter sequence (first 96 frames) and the sun shadow setup from cb_g_Frame
            static LONG64 s_diagN;
            LONG64 dn = s_diagN++;
            if (dn < 96)
                LOGI("ShadowDiag jitter #%I64d: (%.5f, %.5f)", dn, tc.subPixelJitter[0], tc.subPixelJitter[1]);
            if (cb[1] && (dn == 10 || dn == 60 || dn % 600 == 300))
            {
                WatchBuffer(cb[1]);
                float f[92 * 4] = {};
                if (ReadShadow(cb[1], ctx, cbFirst[1] * 16, f, sizeof(f), nullptr))
                {
                    auto V = [&](int r) { return f + r * 4; };
                    LOGI("ShadowDiag sun: mapSize %.0f x %.0f (inv %.6f %.6f)", V(4)[0], V(4)[1], V(4)[2], V(4)[3]);
                    for (int i = 0; i < 4; i++)
                        LOGI("ShadowDiag cascade %d: offset (%.5f %.5f %.5f %.5f) scale (%.5f %.5f %.5f %.5f) splitBlend (%.4f %.4f %.4f %.4f)",
                             i, V(5 + i)[0], V(5 + i)[1], V(5 + i)[2], V(5 + i)[3], V(9 + i)[0], V(9 + i)[1], V(9 + i)[2],
                             V(9 + i)[3], V(13 + i)[0], V(13 + i)[1], V(13 + i)[2], V(13 + i)[3]);
                    LOGI("ShadowDiag noiseScale (%.4f %.4f %.4f %.4f) nearFar (%.2f %.2f %.2f %.2f) fade (%.3f %.3f %.3f %.3f)",
                         V(17)[0], V(17)[1], V(17)[2], V(17)[3], V(18)[0], V(18)[1], V(18)[2], V(18)[3], V(19)[0], V(19)[1],
                         V(19)[2], V(19)[3]);
                    LOGI("ShadowDiag rangesMax (%.4f %.4f %.4f %.4f) rangesMin (%.4f %.4f %.4f %.4f) globalFade (%.2f %.2f %.2f %.2f) "
                         "splitUV (%.4f %.4f %.4f %.4f)",
                         V(20)[0], V(20)[1], V(20)[2], V(20)[3], V(21)[0], V(21)[1], V(21)[2], V(21)[3], V(22)[0], V(22)[1],
                         V(22)[2], V(22)[3], V(23)[0], V(23)[1], V(23)[2], V(23)[3]);
                    for (int r = 0; r < 4; r++)
                        LOGI("ShadowDiag lightProj row %d: continuous (%.6f %.6f %.6f %.6f) quantized (%.6f %.6f %.6f %.6f)", r,
                             V(24 + r)[0], V(24 + r)[1], V(24 + r)[2], V(24 + r)[3], V(28 + r)[0], V(28 + r)[1],
                             V(28 + r)[2], V(28 + r)[3]);
                    const uint32_t* u = (const uint32_t*)f;
                    LOGI("ShadowDiag fadeOutElev %.3f forcedPCF4 %u dynResRatio (%.4f %.4f %.4f %.4f) LRS (%.6f %.6f %.2f %.6f)",
                         V(36)[0], u[82 * 4], V(90)[0], V(90)[1], V(90)[2], V(90)[3], V(37)[0], V(37)[1], V(37)[2], V(37)[3]);
                }
                else
                    LOGI("ShadowDiag: cb_g_Frame not captured yet");
            }
        }
        if (detail || (s_dlssFrames == 0 && taaIndex < 8))
            LOGI("  TAA constants: full %.1fx%.1f low %.1fx%.1f jitter (%.4f, %.4f) isolation %u,%u,%u,%u hdr %u noMV %u",
                 tc.fullResSize[0], tc.fullResSize[1], tc.lowResSize[0], tc.lowResSize[1], tc.subPixelJitter[0],
                 tc.subPixelJitter[1], tc.isolation[0], tc.isolation[1], tc.isolation[2], tc.isolation[3], tc.hdrEnabled,
                 tc.noMotionVector);
        if (lowW < 16 || lowH < 16 || fullW < 16 || fullH < 16 || lowW > fullW || lowH > fullH || fullW > 16384 ||
            fullH > 16384)
        {
            if (detail)
                LOGE("TAA constants look invalid, skipping");
            goto done;
        }
        if (!InitGpu(ctx))
        {
            s_fatal = true;
            LOGE("GPU init failed - mod disabled");
            goto done;
        }

        bool hdr = tc.hdrEnabled != 0;
        bool depthInv = g_cfg.depthInverted >= 0 ? g_cfg.depthInverted != 0 : (s_camRev != 0.0f);
        int flags = (g_cfg.dilatedMV && s_dilateCS ? 0 : NGX_DLSS_MVLowRes) | (depthInv ? NGX_DLSS_DepthInverted : 0) |
                    (hdr ? (NGX_DLSS_IsHDR | (g_cfg.autoExposure ? NGX_DLSS_AutoExposure : 0)) : 0);

        AcquireSRWLockExclusive(&s_viewLock);
        v = GetViewLocked(lowW, lowH, fullW, fullH, flags);
        if (v)
            job->gen = v->gen;
        ReleaseSRWLockExclusive(&s_viewLock);
        if (!v)
            goto done;

        job->view = v;
        job->hdr = hdr;
        job->reset = tc.isolation[0] == 0;
        job->taaIndex = taaIndex;
        job->qpc = NowQpc();
        job->color = srvColor;
        job->motion = srvMotion;
        job->depth = srvDepth;
        job->resp = srvResp;
        srv[0] = srv[1] = srv[2] = srv[3] = nullptr; // ownership moved to the job

        if (!deferred)
        {
            s_modThread = GetCurrentThreadId();
            bool r = RunJob(ctx, job);
            s_modThread = 0;
            FreeJob(job);
            job = nullptr;
            if (!r)
                goto done;
        }
        else
        {
            // close the commands recorded so far; they run before DLSS when the game submits
            HRESULT hr =
                ORIG(h, kCtx_FinishCommandList, PFN_FinishCommandList)(ctx, TRUE, &job->prefix);
            if (FAILED(hr) || !job->prefix)
            {
                LOGE("FinishCommandList (split) failed 0x%08lX", hr);
                goto done;
            }
            ce->pending[ce->pendingCount++] = job;
            job = nullptr;
            ce->splitState = 1;
            ce->discardedCount = 0;
        }

        RecordResolveDraw(ctx, h, taaPS, v, da);
        ok = true;
        if (taaIndex == 1500 || taaIndex == 4000)
        {
            ID3D11Resource* outRes = nullptr;
            rtv[0]->GetResource(&outRes);
            TraceStart(outRes);
            outRes->Release();
        }
    }

done:
    if (job)
        FreeJob(job);
    for (auto*& s : srv)
        SafeRelease(s);
    for (auto*& r : rtv)
        SafeRelease(r);
    SafeRelease(dsv);
    for (auto*& b : cb)
        SafeRelease(b);
    SafeRelease(ctx1);
    if (!ok)
        InterlockedIncrement64(&s_fallbackFrames);
    return ok;
}

extern volatile LONG64 g_presentCount;
static LONG64 s_lastEvalPresent = -1, s_doubleEvals, s_presentsNoEval, s_lastPresentSeen;
void ModOnPresentEvent()
{
    if (s_evalTrace > 0)
    {
        InterlockedDecrement(&s_evalTrace);
        LOGI("EvalTrace %8.2f ms  PRESENT #%I64d (thread %lu)", QpcToMs(NowQpc() - s_traceT0), g_presentCount, GetCurrentThreadId());
    }
}
static void NoteEval(LONG64 taaIndex)
{
    LONG64 pc = g_presentCount;
    if (pc == s_lastEvalPresent)
    {
        s_doubleEvals++;
        if (s_doubleEvals <= 10)
            LOGI("DLSS evaluated twice without a Present in between (TAA pass #%I64d, present #%I64d)", taaIndex, pc);
    }
    else if (s_lastEvalPresent >= 0 && pc - s_lastEvalPresent > 1)
        s_presentsNoEval += pc - s_lastEvalPresent - 1;
    s_lastEvalPresent = pc;
    if (s_evalTrace > 0)
    {
        InterlockedDecrement(&s_evalTrace);
        LOGI("EvalTrace %8.2f ms  DLSS eval (TAA pass #%I64d, thread %lu)", QpcToMs(NowQpc() - s_traceT0), taaIndex,
             GetCurrentThreadId());
    }
}
static void MaybeLogStats()
{
    uint64_t now = NowQpc();
    if (!s_lastStatQpc)
        s_lastStatQpc = now;
    if (QpcToMs(now - s_lastStatQpc) > 30000.0)
    {
        s_lastStatQpc = now;
        LOGI("Stats: TAA passes %I64d, DLSS evaluations %I64d, game-TAA fallbacks %I64d, cb updates %I64d, DLSS %s | presents "
             "%I64d, double evaluations %I64d, presents without evaluation %I64d",
             s_taaFrames, s_dlssFrames, s_fallbackFrames, s_cbUpdates, s_userEnabled ? "on" : "off (toggled)", g_presentCount,
             s_doubleEvals, s_presentsNoEval);
        char cs[160];
        SlCamStats(cs, sizeof(cs));
        LOGI("%s", cs);
    }
}

// ------------------------------------------------------------------ hooks
static void HookDeviceContexts(ID3D11Device* dev)
{
    if (s_lastHookedDev == dev)
        return;
    s_lastHookedDev = dev;
    ID3D11DeviceContext* imm = nullptr;
    dev->GetImmediateContext(&imm);
    if (imm)
    {
        HookContextVtable(imm);
        imm->Release();
    }
}

// Sun shadow mask pass (fullscreen PS: depth + cascades + cloud/long-range shadows -> shadow mask),
// tracked for the shadow debug view.
static ID3D11PixelShader* volatile s_shadowPS[8];
static volatile LONG s_shadowPSCount;
static ID3D11Resource* volatile s_shadowMaskRes;
static bool IsShadowMaskPS(ID3D11PixelShader* ps)
{
    LONG n = s_shadowPSCount;
    for (LONG i = 0; i < n && i < 8; i++)
        if (s_shadowPS[i] == ps)
            return true;
    return false;
}
static void CaptureShadowMask(ID3D11DeviceContext* ctx)
{
    ID3D11RenderTargetView* rtv = nullptr;
    ctx->OMGetRenderTargets(1, &rtv, nullptr);
    if (!rtv)
        return;
    ID3D11Resource* res = nullptr;
    rtv->GetResource(&res);
    if (res && res != s_shadowMaskRes)
    {
        ID3D11Device* dev = nullptr;
        ctx->GetDevice(&dev);
        ID3D11ShaderResourceView* srv = nullptr;
        if (dev && SUCCEEDED(dev->CreateShaderResourceView(res, nullptr, &srv)))
        {
            D3D11_TEXTURE2D_DESC td = {};
            ((ID3D11Texture2D*)res)->GetDesc(&td);
            LOGI("Shadow mask render target %p: %ux%u fmt %u", res, td.Width, td.Height, td.Format);
            ID3D11ShaderResourceView* old = (ID3D11ShaderResourceView*)InterlockedExchangePointer((void* volatile*)&s_shadowMaskSrv, srv);
            s_shadowMaskRes = res; // the SRV keeps the texture alive
            if (old)
                old->Release();
        }
        SafeRelease(dev);
    }
    SafeRelease(res);
    rtv->Release();
}

static HRESULT STDMETHODCALLTYPE Hk_CreatePixelShader(ID3D11Device* dev, const void* code, SIZE_T len,
                                                      ID3D11ClassLinkage* link, ID3D11PixelShader** out)
{
    HRESULT hr = o_CreatePixelShader(dev, code, len, link, out);
    HookDeviceContexts(dev);
    if (SUCCEEDED(hr) && out && *out && code && len > 64 && MemFind(code, len, "t_g_LRS_TemporalClipTexture") &&
        MemFind(code, len, "t_DepthSurface") && MemFind(code, len, "cb_g_VolumeCloudsShadow") &&
        !MemFind(code, len, "t_BRDFLUT") && !MemFind(code, len, "t_g_FogSampler"))
    {
        LONG i = InterlockedIncrement(&s_shadowPSCount) - 1;
        if (i < 8)
        {
            s_shadowPS[i] = *out;
            LOGI("Sun shadow mask pixel shader detected: %p (%Iu bytes)", *out, (size_t)len);
        }
    }
    if (SUCCEEDED(hr) && out && *out && code)
        RegisterShaderName(*out, code, len);
    if (SUCCEEDED(hr) && out && *out && code && len > 64 && MemFind(code, len, "t_TemporalAA_ColorAccumulation") &&
        MemFind(code, len, "cb_g_TemporalAA_Constants"))
    {
        LONG i = InterlockedIncrement(&s_taaCount) - 1;
        if (i < 64)
        {
            s_taaPS[i] = *out;
            LOGI("TAA pixel shader detected: %p (%Iu bytes) [%ld]", *out, (size_t)len, i);
        }
        else
            InterlockedDecrement(&s_taaCount);
    }
    return hr;
}

static HRESULT STDMETHODCALLTYPE Hk_CreateDeferredContext(ID3D11Device* dev, UINT flags, ID3D11DeviceContext** out)
{
    HRESULT hr = o_CreateDeferredContext(dev, flags, out);
    if (SUCCEEDED(hr) && out && *out)
        HookContextVtable(*out);
    return hr;
}

static void STDMETHODCALLTYPE Hk_PSSetShader(ID3D11DeviceContext* ctx, ID3D11PixelShader* ps,
                                             ID3D11ClassInstance* const* ci, UINT n)
{
    VtHooks* h = HooksFor(ctx);
    if (s_taaCount || s_shadowPSCount)
        GetCtx(ctx)->ps = ps;
    ORIG(h, kCtx_PSSetShader, PFN_PSSetShader)(ctx, ps, ci, n);
}

static inline CtxInfo* TaaCandidate(ID3D11DeviceContext* ctx)
{
    if (!s_taaCount || s_modThread == GetCurrentThreadId())
        return nullptr;
    LONG n = s_ctxCount;
    for (LONG i = 0; i < n; i++)
        if (s_ctx[i].ctx == ctx)
            return IsTaaPS(s_ctx[i].ps) ? &s_ctx[i] : nullptr;
    return nullptr;
}

static bool TryTaa(ID3D11DeviceContext* ctx, VtHooks* h, CtxInfo* ce, const DrawArgs& da)
{
    ID3D11PixelShader* cur = nullptr;
    ctx->PSGetShader(&cur, nullptr, nullptr);
    if (cur)
        cur->Release();
    if (!IsTaaPS(cur))
    {
        ce->ps = cur;
        return false;
    }
    bool r = HandleTaa(ctx, h, ce, cur, da);
    ce->ps = cur;
    MaybeLogStats();
    return r;
}

static void STDMETHODCALLTYPE Hk_Draw(ID3D11DeviceContext* ctx, UINT vc, UINT sv)
{
    VtHooks* h = HooksFor(ctx);
    if (g_cfg.debugView == 6 && s_shadowPSCount && s_modThread != GetCurrentThreadId())
    {
        CtxInfo* ci = GetCtx(ctx);
        if (ci && IsShadowMaskPS(ci->ps))
            CaptureShadowMask(ctx);
    }
    if (s_traceActive)
        TraceDraw(ctx, false);
    if (CtxInfo* ce = TaaCandidate(ctx))
    {
        DrawArgs da = {0, vc, sv, 0, 0, 0};
        if (TryTaa(ctx, h, ce, da))
            return;
    }
    ORIG(h, kCtx_Draw, PFN_Draw)(ctx, vc, sv);
}

static void STDMETHODCALLTYPE Hk_DrawIndexed(ID3D11DeviceContext* ctx, UINT ic, UINT si, INT bv)
{
    VtHooks* h = HooksFor(ctx);
    if (s_traceActive)
        TraceDraw(ctx, false);
    if (CtxInfo* ce = TaaCandidate(ctx))
    {
        DrawArgs da = {1, ic, si, 0, 0, bv};
        if (TryTaa(ctx, h, ce, da))
            return;
    }
    ORIG(h, kCtx_DrawIndexed, PFN_DrawIndexed)(ctx, ic, si, bv);
}

static void STDMETHODCALLTYPE Hk_DrawInstanced(ID3D11DeviceContext* ctx, UINT vc, UINT inst, UINT sv, UINT si)
{
    VtHooks* h = HooksFor(ctx);
    if (s_traceActive)
        TraceDraw(ctx, false);
    if (CtxInfo* ce = TaaCandidate(ctx))
    {
        DrawArgs da = {2, vc, inst, sv, si, 0};
        if (TryTaa(ctx, h, ce, da))
            return;
    }
    ORIG(h, kCtx_DrawInstanced, PFN_DrawInstanced)(ctx, vc, inst, sv, si);
}

static void STDMETHODCALLTYPE Hk_DrawIndexedInstanced(ID3D11DeviceContext* ctx, UINT ic, UINT inst, UINT si, INT bv,
                                                      UINT sinst)
{
    VtHooks* h = HooksFor(ctx);
    if (s_traceActive)
        TraceDraw(ctx, false);
    if (CtxInfo* ce = TaaCandidate(ctx))
    {
        DrawArgs da = {3, ic, inst, si, sinst, bv};
        if (TryTaa(ctx, h, ce, da))
            return;
    }
    ORIG(h, kCtx_DrawIndexedInstanced, PFN_DrawIndexedInstanced)(ctx, ic, inst, si, bv, sinst);
}

static void STDMETHODCALLTYPE Hk_Dispatch(ID3D11DeviceContext* ctx, UINT x, UINT y, UINT z)
{
    VtHooks* h = HooksFor(ctx);
    if (s_traceActive && s_modThread != GetCurrentThreadId())
        TraceDraw(ctx, true);
    ORIG(h, kCtx_Dispatch, PFN_Dispatch)(ctx, x, y, z);
}

void ProbeViewport(void* ret, ID3D11DeviceContext* ctx, UINT n, const D3D11_VIEWPORT* vp);
static void STDMETHODCALLTYPE Hk_RSSetViewports(ID3D11DeviceContext* ctx, UINT n, const D3D11_VIEWPORT* vp)
{
    VtHooks* h = HooksFor(ctx);
    if (g_cfg.devCommands)
        ProbeViewport(__builtin_return_address(0), ctx, n, vp);
    ORIG(h, kCtx_RSSetViewports, PFN_RSSetViewports)(ctx, n, vp);
}

// The game's swap chain is found from its back buffer when a render target view is created for it
// (we no longer hook DXGI factories: overlays such as Ubisoft Connect's re-hook those and the two
// hooks could end up calling each other until the game crashed at startup).
typedef HRESULT(STDMETHODCALLTYPE* PFN_CreateRTV)(ID3D11Device*, ID3D11Resource*, const D3D11_RENDER_TARGET_VIEW_DESC*,
                                                  ID3D11RenderTargetView**);
static PFN_CreateRTV o_CreateRTV;
static ID3D11Device* volatile s_gameCreatedDev; // device created by GRB.exe itself
void OverlayOnRenderTarget(ID3D11Device* dev, ID3D11Resource* res);
bool ModIsGameDevice(ID3D11Device* dev)
{
    ID3D11Device* g = s_gameCreatedDev;
    return !g || g == dev;
}
static HRESULT STDMETHODCALLTYPE Hk_CreateRTV(ID3D11Device* dev, ID3D11Resource* res, const D3D11_RENDER_TARGET_VIEW_DESC* d,
                                              ID3D11RenderTargetView** out)
{
    HRESULT hr = o_CreateRTV(dev, res, d, out);
    if (SUCCEEDED(hr) && res && out && *out)
        OverlayOnRenderTarget(dev, res);
    return hr;
}

static HRESULT STDMETHODCALLTYPE Hk_CreateComputeShader(ID3D11Device* dev, const void* code, SIZE_T len,
                                                        ID3D11ClassLinkage* link, ID3D11ComputeShader** out)
{
    HRESULT hr = o_CreateComputeShader(dev, code, len, link, out);
    if (SUCCEEDED(hr) && out && *out && code)
        RegisterShaderName(*out, code, len);
    return hr;
}

static volatile LONG s_convertedMaps;

static HRESULT STDMETHODCALLTYPE Hk_Map(ID3D11DeviceContext* ctx, ID3D11Resource* r, UINT sub, D3D11_MAP type,
                                        UINT flags, D3D11_MAPPED_SUBRESOURCE* m)
{
    VtHooks* h = HooksFor(ctx);

    // After we split a deferred command list, the new list has not seen the DISCARD that a
    // NO_OVERWRITE map relies on; promote the first such map of each resource to DISCARD.
    if (type == D3D11_MAP_WRITE_NO_OVERWRITE && s_ctxCount)
    {
        LONG n = s_ctxCount;
        for (LONG i = 0; i < n; i++)
            if (s_ctx[i].ctx == ctx)
            {
                CtxInfo& ce = s_ctx[i];
                if (ce.splitState)
                {
                    bool seen = false;
                    for (int k = 0; k < ce.discardedCount; k++)
                        if (ce.discarded[k] == r)
                            seen = true;
                    if (!seen)
                    {
                        type = D3D11_MAP_WRITE_DISCARD;
                        if (ce.discardedCount < 64)
                            ce.discarded[ce.discardedCount++] = r;
                        if (InterlockedIncrement(&s_convertedMaps) <= 5)
                            LOGI("Promoted NO_OVERWRITE map of %p to DISCARD after a command-list split", r);
                    }
                }
                break;
            }
    }

    HRESULT hr = ORIG(h, kCtx_Map, PFN_Map)(ctx, r, sub, type, flags, m);

    if (type == D3D11_MAP_WRITE_DISCARD && s_ctxCount && hr == S_OK)
    {
        LONG n = s_ctxCount;
        for (LONG i = 0; i < n; i++)
            if (s_ctx[i].ctx == ctx)
            {
                CtxInfo& ce = s_ctx[i];
                if (ce.splitState && ce.discardedCount < 64)
                {
                    bool seen = false;
                    for (int k = 0; k < ce.discardedCount; k++)
                        if (ce.discarded[k] == r)
                            seen = true;
                    if (!seen)
                        ce.discarded[ce.discardedCount++] = r;
                }
                break;
            }
    }

    if (s_watchCount && hr == S_OK && m && m->pData && type != D3D11_MAP_READ)
    {
        int w = FindWatch(r);
        if (w >= 0)
        {
            UINT bw = s_watch[w].byteWidth;
            if (bw > kSmallCB)
                ShadowPointer(r, ctx, bw, m->pData);
            else
            {
                AcquireSRWLockExclusive(&s_pendingLock);
                LONG n = s_pendingCount;
                int slot = -1;
                for (LONG k = 0; k < n; k++)
                    if (s_pending[k].ctx == ctx && s_pending[k].res == r && s_pending[k].sub == sub)
                        slot = k;
                if (slot < 0 && n < 64)
                    slot = n, s_pendingCount = n + 1;
                if (slot >= 0)
                    s_pending[slot] = {ctx, r, sub, m->pData, bw};
                ReleaseSRWLockExclusive(&s_pendingLock);
            }
        }
    }
    return hr;
}

static void STDMETHODCALLTYPE Hk_Unmap(ID3D11DeviceContext* ctx, ID3D11Resource* r, UINT sub)
{
    VtHooks* h = HooksFor(ctx);
    if (s_pendingCount)
    {
        PendingMap pm = {};
        bool found = false;
        AcquireSRWLockExclusive(&s_pendingLock);
        LONG n = s_pendingCount;
        for (LONG k = 0; k < n; k++)
            if (s_pending[k].ctx == ctx && s_pending[k].res == r && s_pending[k].sub == sub)
            {
                pm = s_pending[k];
                s_pending[k] = s_pending[n - 1];
                s_pendingCount = n - 1;
                found = true;
                break;
            }
        ReleaseSRWLockExclusive(&s_pendingLock);
        if (found)
            ShadowWrite(r, ctx, pm.bytes, 0, pm.ptr, pm.bytes);
    }
    ORIG(h, kCtx_Unmap, PFN_Unmap)(ctx, r, sub);
}

static void STDMETHODCALLTYPE Hk_UpdateSubresource(ID3D11DeviceContext* ctx, ID3D11Resource* r, UINT sub,
                                                   const D3D11_BOX* box, const void* src, UINT rp, UINT dp)
{
    VtHooks* h = HooksFor(ctx);
    if (s_watchCount && src)
    {
        int w = FindWatch(r);
        if (w >= 0)
        {
            UINT off = box ? box->left : 0;
            UINT bytes = box ? box->right - box->left : s_watch[w].byteWidth;
            ShadowWrite(r, ctx, s_watch[w].byteWidth, off, src, bytes);
        }
    }
    ORIG(h, kCtx_UpdateSubresource, PFN_UpdateSubresource)(ctx, r, sub, box, src, rp, dp);
}

static void STDMETHODCALLTYPE Hk_UpdateSubresource1(ID3D11DeviceContext* ctx, ID3D11Resource* r, UINT sub,
                                                    const D3D11_BOX* box, const void* src, UINT rp, UINT dp, UINT fl)
{
    VtHooks* h = HooksFor(ctx);
    if (s_watchCount && src)
    {
        int w = FindWatch(r);
        if (w >= 0)
        {
            UINT off = box ? box->left : 0;
            UINT bytes = box ? box->right - box->left : s_watch[w].byteWidth;
            ShadowWrite(r, ctx, s_watch[w].byteWidth, off, src, bytes);
        }
    }
    ORIG(h, kCtx1_UpdateSubresource1, PFN_UpdateSubresource1)(ctx, r, sub, box, src, rp, dp, fl);
}

static void STDMETHODCALLTYPE Hk_CopySubresourceRegion(ID3D11DeviceContext* ctx, ID3D11Resource* dst, UINT ds, UINT x,
                                                       UINT y, UINT z, ID3D11Resource* src, UINT ss, const D3D11_BOX* box)
{
    VtHooks* h = HooksFor(ctx);
    if (s_watchCount && FindWatch(dst) >= 0)
        InterlockedIncrement64(&s_cbGpuCopies);
    ORIG(h, kCtx_CopySubresourceRegion, PFN_CopySubresourceRegion)(ctx, dst, ds, x, y, z, src, ss, box);
}

static void STDMETHODCALLTYPE Hk_CopyResource(ID3D11DeviceContext* ctx, ID3D11Resource* dst, ID3D11Resource* src)
{
    VtHooks* h = HooksFor(ctx);
    if (s_watchCount && FindWatch(dst) >= 0)
        InterlockedIncrement64(&s_cbGpuCopies);
    ORIG(h, kCtx_CopyResource, PFN_CopyResource)(ctx, dst, src);
}

static HRESULT STDMETHODCALLTYPE Hk_FinishCommandList(ID3D11DeviceContext* ctx, BOOL restore, ID3D11CommandList** out)
{
    VtHooks* h = HooksFor(ctx);
    if (ID3D11DeviceContext* imm = s_immCtx)
    {
        void** ivt = *(void***)imm;
        if (ivt[kCtx_ExecuteCommandList] != (void*)Hk_ExecuteCommandList)
        {
            static volatile LONG warned;
            if (InterlockedIncrement(&warned) <= 5)
                LOGI("Immediate context vtable %p lost our hooks (rewritten by D3D11/another tool) - re-hooking", ivt);
            HookOneVtable(imm, "re-hook");
        }
    }
    HRESULT hr = ORIG(h, kCtx_FinishCommandList, PFN_FinishCommandList)(ctx, restore, out);
    if (!s_taaCount)
        return hr;
    CtxInfo* ce = GetCtx(ctx);
    ce->splitState = 0;
    ce->discardedCount = 0;
    if (!ce->pendingCount)
        return hr;
    if (FAILED(hr) || !out || !*out)
    {
        for (int i = 0; i < ce->pendingCount; i++)
            FreeJob(ce->pending[i]);
        ce->pendingCount = 0;
        return hr;
    }
    // attach the jobs to the command list the game will submit
    AcquireSRWLockExclusive(&s_clLock);
    ClJobs* slot = nullptr;
    uint64_t now = NowQpc();
    for (ClJobs& c : s_clJobs)
        if (!c.cl)
        {
            slot = &c;
            break;
        }
    if (!slot)
    {
        // evict the oldest (never submitted) entry
        slot = &s_clJobs[0];
        for (ClJobs& c : s_clJobs)
            if (c.qpc < slot->qpc)
                slot = &c;
        for (int i = 0; i < slot->count; i++)
            FreeJob(slot->jobs[i]);
        SafeRelease(slot->cl);
        LOGI("Dropped DLSS jobs of a command list that was never submitted");
    }
    slot->cl = *out;
    slot->cl->AddRef();
    slot->count = ce->pendingCount;
    for (int i = 0; i < ce->pendingCount; i++)
        slot->jobs[i] = ce->pending[i];
    slot->qpc = now;
    ce->pendingCount = 0;
    {
        static volatile LONG logged;
        if (InterlockedIncrement(&logged) <= 3)
            LOGI("FinishCommandList on %p -> cl %p carries %d DLSS job(s)", ctx, *out, slot->count);
    }
    ReleaseSRWLockExclusive(&s_clLock);
    return hr;
}

static void STDMETHODCALLTYPE Hk_ExecuteCommandList(ID3D11DeviceContext* ctx, ID3D11CommandList* cl, BOOL restore)
{
    VtHooks* h = HooksFor(ctx);
    if (!s_immCtx && ctx->GetType() == D3D11_DEVICE_CONTEXT_IMMEDIATE)
    {
        ctx->AddRef();
        s_immCtx = ctx;
    }
    ClJobs found = {};
    if (cl && s_taaCount)
    {
        AcquireSRWLockExclusive(&s_clLock);
        for (ClJobs& c : s_clJobs)
            if (c.cl == cl)
            {
                found = c;
                c.cl = nullptr;
                c.count = 0;
                break;
            }
        ReleaseSRWLockExclusive(&s_clLock);
    }
    if (found.cl && ctx->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE)
    {
        static bool once;
        if (!once)
            LOGE("Command list with DLSS work executed on a deferred context (nested) - not supported yet");
        once = true;
        for (int i = 0; i < found.count; i++)
        {
            // keep the prefix ordering correct at least: execute it into the outer list
            ORIG(h, kCtx_ExecuteCommandList, PFN_ExecuteCommandList)(ctx, found.jobs[i]->prefix, TRUE);
            FreeJob(found.jobs[i]);
        }
        found.cl->Release();
        found.cl = nullptr;
        found.count = 0;
    }
    if (found.cl)
    {
        for (int i = 0; i < found.count; i++)
        {
            Job* j = found.jobs[i];
            ORIG(h, kCtx_ExecuteCommandList, PFN_ExecuteCommandList)(ctx, j->prefix, TRUE);
            s_modThread = GetCurrentThreadId();
            if (!RunJob(ctx, j))
                InterlockedIncrement64(&s_fallbackFrames);
            s_modThread = 0;
            FreeJob(j);
        }
        found.cl->Release(); // our reference; the game still holds its own
    }
    {
        static volatile LONG logged;
        if (s_taaCount && InterlockedIncrement(&logged) <= 6)
            LOGI("ExecuteCommandList on %p (%s) cl %p restore %d jobs %d", ctx,
                 ctx->GetType() == D3D11_DEVICE_CONTEXT_IMMEDIATE ? "immediate" : "deferred", cl, restore, found.count);
    }
    ORIG(h, kCtx_ExecuteCommandList, PFN_ExecuteCommandList)(ctx, cl, restore);
    if (!restore && s_taaCount)
        GetCtx(ctx)->ps = nullptr;
}

static void STDMETHODCALLTYPE Hk_ClearState(ID3D11DeviceContext* ctx)
{
    VtHooks* h = HooksFor(ctx);
    ORIG(h, kCtx_ClearState, PFN_ClearState)(ctx);
    if (s_taaCount)
        GetCtx(ctx)->ps = nullptr;
}

// ------------------------------------------------------------------ install
static void LogUnknownCtxVtable(void** vt)
{
    static volatile LONG s_n;
    if (InterlockedIncrement(&s_n) <= 5)
        LOGI("Hook: context call through vtable %p we did not patch (copied by another layer)", vt);
}

void LogCaller(const char* what, void* ret, void* obj)
{
    HMODULE m = nullptr;
    char path[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)ret, &m) && m)
        GetModuleFileNameA(m, path, MAX_PATH);
    const char* base = path;
    for (const char* p = path; *p; p++)
        if (*p == '\\' || *p == '/')
            base = p + 1;
    LOGI("%s: called from %s+0x%llX (object %p, thread %lu)", what, base,
         m ? (unsigned long long)((uint8_t*)ret - (uint8_t*)m) : 0ull, obj, GetCurrentThreadId());
}

static bool PatchSlot(void** vt, int idx, void* hook, void** orig)
{
    if (vt[idx] == hook)
        return true;
    // Already hooked by us, and another layer (OptiScaler, an overlay, the driver) has since put its own
    // hook in front of ours: that hook calls ours, so leave it. Re-patching would save their hook as our
    // "original" and the two hooks would call each other forever (stack overflow at startup).
    if (*orig && vt[idx] != *orig)
        return true;
    DWORD old;
    if (!VirtualProtect(&vt[idx], sizeof(void*), PAGE_READWRITE, &old))
    {
        if (!VirtualProtect(&vt[idx], sizeof(void*), PAGE_EXECUTE_READWRITE, &old))
            return false;
    }
    *orig = vt[idx];
    vt[idx] = hook;
    DWORD tmp;
    VirtualProtect(&vt[idx], sizeof(void*), old, &tmp);
    return true;
}

static void HookOneVtable(ID3D11DeviceContext* ctx, const char* iface)
{
    void** vt = *(void***)ctx;
    if (vt[kCtx_Draw] == (void*)Hk_Draw && vt[kCtx_ExecuteCommandList] == (void*)Hk_ExecuteCommandList)
        return;
    AcquireSRWLockExclusive(&s_hookLock);
    if (vt[kCtx_Draw] != (void*)Hk_Draw || vt[kCtx_ExecuteCommandList] != (void*)Hk_ExecuteCommandList)
    {
        ID3D11DeviceContext1* c1 = nullptr;
        ctx->QueryInterface(__uuidof(ID3D11DeviceContext1), (void**)&c1);
        LONG n = s_ctxHookCount;
        VtHooks* h = nullptr;
        for (LONG i = 0; i < n; i++)
            if (s_ctxHooks[i].vt == vt)
                h = &s_ctxHooks[i];
        bool fresh = false;
        if (!h && n < 16)
        {
            h = &s_ctxHooks[n];
            fresh = true;
        }
        if (h)
        {
            h->vt = vt;
            // publish the record before any slot points at our hooks, so a call that races the
            // patching finds this vtable's originals (not another device's)
            MemoryBarrier();
            if (fresh)
            {
                s_ctxHookCount = n + 1;
                fresh = false;
            }
            MemoryBarrier();
            PatchSlot(vt, kCtx_PSSetShader, (void*)Hk_PSSetShader, &h->orig[kCtx_PSSetShader]);
            PatchSlot(vt, kCtx_Map, (void*)Hk_Map, &h->orig[kCtx_Map]);
            PatchSlot(vt, kCtx_Unmap, (void*)Hk_Unmap, &h->orig[kCtx_Unmap]);
            PatchSlot(vt, kCtx_UpdateSubresource, (void*)Hk_UpdateSubresource, &h->orig[kCtx_UpdateSubresource]);
            PatchSlot(vt, kCtx_ExecuteCommandList, (void*)Hk_ExecuteCommandList, &h->orig[kCtx_ExecuteCommandList]);
            PatchSlot(vt, kCtx_FinishCommandList, (void*)Hk_FinishCommandList, &h->orig[kCtx_FinishCommandList]);
            PatchSlot(vt, kCtx_CopySubresourceRegion, (void*)Hk_CopySubresourceRegion,
                      &h->orig[kCtx_CopySubresourceRegion]);
            PatchSlot(vt, kCtx_CopyResource, (void*)Hk_CopyResource, &h->orig[kCtx_CopyResource]);
            PatchSlot(vt, kCtx_ClearState, (void*)Hk_ClearState, &h->orig[kCtx_ClearState]);
            if (c1)
                PatchSlot(vt, kCtx1_UpdateSubresource1, (void*)Hk_UpdateSubresource1, &h->orig[kCtx1_UpdateSubresource1]);
            PatchSlot(vt, kCtx_DrawIndexed, (void*)Hk_DrawIndexed, &h->orig[kCtx_DrawIndexed]);
            PatchSlot(vt, kCtx_Dispatch, (void*)Hk_Dispatch, &h->orig[kCtx_Dispatch]);
            if (g_cfg.devCommands)
                PatchSlot(vt, kCtx_RSSetViewports, (void*)Hk_RSSetViewports, &h->orig[kCtx_RSSetViewports]);
            PatchSlot(vt, kCtx_DrawInstanced, (void*)Hk_DrawInstanced, &h->orig[kCtx_DrawInstanced]);
            PatchSlot(vt, kCtx_DrawIndexedInstanced, (void*)Hk_DrawIndexedInstanced, &h->orig[kCtx_DrawIndexedInstanced]);
            MemoryBarrier();
            if (fresh)
                s_ctxHookCount = n + 1;
            MemoryBarrier();
            PatchSlot(vt, kCtx_Draw, (void*)Hk_Draw, &h->orig[kCtx_Draw]); // last: the "hooked" marker
            LOGI("Context %p [%s] (%s) vtable %p hooked", ctx, iface,
                 ctx->GetType() == D3D11_DEVICE_CONTEXT_IMMEDIATE ? "immediate" : "deferred", vt);
        }
        else
            LOGE("Too many context vtables, %p not hooked", vt);
        SafeRelease(c1);
    }
    ReleaseSRWLockExclusive(&s_hookLock);
}

// Some D3D11 layers hand out a separate object (with its own vtable) per interface version, so
// hook every interface the game might call through.
static void HookContextVtable(ID3D11DeviceContext* ctx)
{
    HookOneVtable(ctx, "ID3D11DeviceContext");
    struct
    {
        const IID* iid;
        const char* name;
    } ifs[] = {{&__uuidof(ID3D11DeviceContext), "ID3D11DeviceContext(QI)"},
               {&__uuidof(ID3D11DeviceContext1), "ID3D11DeviceContext1"},
               {&__uuidof(ID3D11DeviceContext2), "ID3D11DeviceContext2"},
               {&__uuidof(ID3D11DeviceContext3), "ID3D11DeviceContext3"},
               {&__uuidof(ID3D11DeviceContext4), "ID3D11DeviceContext4"}};
    for (auto& i : ifs)
    {
        ID3D11DeviceContext* c = nullptr;
        if (SUCCEEDED(ctx->QueryInterface(*i.iid, (void**)&c)) && c)
        {
            if (c != ctx)
                LOGD("  %s of %p is a different object %p", i.name, ctx, c);
            HookOneVtable(c, i.name);
            c->Release();
        }
    }
}


// ------------------------------------------------------------------ feature-level compatibility
// OptiScaler upgrades the game's D3D_FEATURE_LEVEL_11_0 device request to 11_1 (for its DX12 bridge).
// Breakpoint then aborts with "DirectX error". We hook the game's own import of D3D11CreateDevice and
// report back the feature level it asked for (the device itself stays 11_1).
typedef HRESULT(WINAPI* PFN_GetFeatureLevel)(ID3D11Device*);
static PFN_D3D11_CREATE_DEVICE o_GameD3D11CreateDevice;
static PFN_D3D11_CREATE_DEVICE_AND_SWAP_CHAIN o_GameD3D11CreateDeviceAndSwapChain;
static D3D_FEATURE_LEVEL (STDMETHODCALLTYPE* o_GetFeatureLevel)(ID3D11Device*);
static ID3D11Device* volatile s_maskDev[8];
static D3D_FEATURE_LEVEL s_maskFl[8];

static bool CallerIsUpscalerStack(void* ret)
{
    HMODULE m = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)ret, &m) ||
        !m)
        return false;
    // OptiScaler, NGX core/snippets and ourselves see the real feature level
    return GetProcAddress(m, "NVSDK_NGX_D3D11_Init") || GetProcAddress(m, "NVSDK_NGX_D3D12_Init") ||
           GetProcAddress(m, "GRB_DLSS_Version") || GetProcAddress(m, "NVSDK_NGX_D3D11_EvaluateFeature_Internal");
}

static D3D_FEATURE_LEVEL STDMETHODCALLTYPE Hk_GetFeatureLevel(ID3D11Device* dev)
{
    D3D_FEATURE_LEVEL fl = o_GetFeatureLevel(dev);
    if (CallerIsUpscalerStack(__builtin_return_address(0)))
        return fl;
    for (int i = 0; i < 8; i++)
        if (s_maskDev[i] == dev && fl > s_maskFl[i])
            return s_maskFl[i];
    return fl;
}

static D3D_FEATURE_LEVEL RequestedMax(const D3D_FEATURE_LEVEL* fls, UINT n)
{
    if (!fls || !n)
        return D3D_FEATURE_LEVEL_11_0; // default list tops out at 11_0
    D3D_FEATURE_LEVEL m = (D3D_FEATURE_LEVEL)0;
    for (UINT i = 0; i < n; i++)
        if (fls[i] > m)
            m = fls[i];
    return m;
}

static void MaskDevice(ID3D11Device* dev, D3D_FEATURE_LEVEL want, D3D_FEATURE_LEVEL* outFl)
{
    if (!dev || !o_GetFeatureLevel)
        return;
    // Call through dev's own vtable, not o_GetFeatureLevel: that is the original from our temporary
    // device's vtable, which under ReShade is ReShade's proxy device class. With OptiScaler in front of
    // ReShade, d3d11 hands us raw (unwrapped) devices here, and calling the proxy method on one crashes.
    // If dev does use the hooked vtable, Hk_GetFeatureLevel sees us as the caller and reports the real level.
    D3D_FEATURE_LEVEL got = dev->GetFeatureLevel();
    if (got <= want)
        return;
    for (int i = 0; i < 8; i++)
        if (!s_maskDev[i])
        {
            s_maskFl[i] = want;
            s_maskDev[i] = dev;
            break;
        }
    if (outFl)
        *outFl = want;
    LOGI("Device %p was created at feature level 0x%X but the game asked for 0x%X: reporting 0x%X to the game", dev, got,
         want, want);
}

static HRESULT WINAPI Hk_GameD3D11CreateDevice(IDXGIAdapter* a, D3D_DRIVER_TYPE t, HMODULE sw, UINT fl,
                                               const D3D_FEATURE_LEVEL* fls, UINT n, UINT sdk, ID3D11Device** dev,
                                               D3D_FEATURE_LEVEL* outFl, ID3D11DeviceContext** ctx)
{
    D3D_FEATURE_LEVEL want = RequestedMax(fls, n);
    static volatile LONG s_logs;
    bool log = InterlockedIncrement(&s_logs) <= 12;
    if (log)
        LogCaller("D3D11CreateDevice begin", __builtin_return_address(0), nullptr);
    HRESULT hr = o_GameD3D11CreateDevice(a, t, sw, fl, fls, n, sdk, dev, outFl, ctx);
    if (log)
        LOGI("D3D11CreateDevice end hr=0x%08lX dev=%p", hr, dev ? *dev : nullptr);
    if (SUCCEEDED(hr) && dev && *dev)
    {
        MaskDevice(*dev, want, outFl);
        HMODULE caller = nullptr;
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCWSTR)__builtin_return_address(0), &caller) &&
            caller == GetModuleHandleW(nullptr))
            s_gameCreatedDev = *dev;
    }
    return hr;
}

static HRESULT WINAPI Hk_GameD3D11CreateDeviceAndSwapChain(IDXGIAdapter* a, D3D_DRIVER_TYPE t, HMODULE sw, UINT fl,
                                                           const D3D_FEATURE_LEVEL* fls, UINT n, UINT sdk,
                                                           const DXGI_SWAP_CHAIN_DESC* sd, IDXGISwapChain** sc,
                                                           ID3D11Device** dev, D3D_FEATURE_LEVEL* outFl,
                                                           ID3D11DeviceContext** ctx)
{
    D3D_FEATURE_LEVEL want = RequestedMax(fls, n);
    LogCaller("D3D11CreateDeviceAndSwapChain begin", __builtin_return_address(0), nullptr);
    HRESULT hr = o_GameD3D11CreateDeviceAndSwapChain(a, t, sw, fl, fls, n, sdk, sd, sc, dev, outFl, ctx);
    LOGI("D3D11CreateDeviceAndSwapChain end hr=0x%08lX", hr);
    if (SUCCEEDED(hr) && dev && *dev)
        MaskDevice(*dev, want, outFl);
    return hr;
}


// Chain in front of an existing inline detour (OptiScaler's) on a d3d11 export: the export starts
// with a jmp into the detour; we redirect that jmp to our hook and call the old destination.
static void* JmpTarget(uint8_t* p)
{
    if (p[0] == 0xE9)
        return p + 5 + *(int32_t*)(p + 1);
    if (p[0] == 0xFF && p[1] == 0x25)
        return *(void**)(p + 6 + *(int32_t*)(p + 2));
    if (p[0] == 0x48 && p[1] == 0xB8 && p[10] == 0xFF && p[11] == 0xE0)
        return *(void**)(p + 2);
    return nullptr;
}

uint8_t* AllocNear(uint8_t* addr)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    uintptr_t gran = si.dwAllocationGranularity;
    uintptr_t base = ((uintptr_t)addr) & ~(gran - 1);
    for (uintptr_t d = gran; d < 0x70000000; d += gran)
    {
        uint8_t* tryAddrs[2] = {(uint8_t*)(base - d), (uint8_t*)(base + d)};
        for (uint8_t* t : tryAddrs)
        {
            void* m = VirtualAlloc(t, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
            if (m)
                return (uint8_t*)m;
        }
    }
    return nullptr;
}

bool ChainExport(HMODULE mod, const char* name, void* hook, void** orig)
{
    uint8_t* fn = (uint8_t*)GetProcAddress(mod, name);
    if (!fn)
    {
        LOGI("ChainExport: %s not found in %p", name, mod);
        return false;
    }
    LOGI("ChainExport: %s at %p bytes %02X %02X %02X %02X %02X %02X", name, fn, fn[0], fn[1], fn[2], fn[3], fn[4], fn[5]);
    void* target = JmpTarget(fn);
    if (!target)
    {
        LOGI("%s is not detoured (no OptiScaler?) - feature-level masking not needed there", name);
        return false;
    }
    uint8_t* stub = AllocNear(fn);
    if (!stub)
    {
        LOGE("ChainExport: no memory near %p", fn);
        return false;
    }
    stub[0] = 0xFF;
    stub[1] = 0x25;
    *(int32_t*)(stub + 2) = 0;
    *(void**)(stub + 6) = hook;
    intptr_t rel = (intptr_t)stub - (intptr_t)(fn + 5);
    if (rel > INT32_MAX || rel < INT32_MIN)
        return false;
    *orig = target;
    DWORD old;
    if (!VirtualProtect(fn, 16, PAGE_EXECUTE_READWRITE, &old))
        return false;
    uint8_t patch[5] = {0xE9};
    *(int32_t*)(patch + 1) = (int32_t)rel;
    memcpy(fn, patch, 5);
    VirtualProtect(fn, 16, old, &old);
    FlushInstructionCache(GetCurrentProcess(), fn, 16);
    LOGI("%s: chained in front of existing detour %p", name, target);
    return true;
}

static void HookGameImports()
{
    uint8_t* base = (uint8_t*)GetModuleHandleW(nullptr);
    IMAGE_DOS_HEADER* dos = (IMAGE_DOS_HEADER*)base;
    IMAGE_NT_HEADERS64* nt = (IMAGE_NT_HEADERS64*)(base + dos->e_lfanew);
    IMAGE_DATA_DIRECTORY dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress)
        return;
    HMODULE d3d11 = GetModuleHandleW(L"d3d11.dll");
    void* realCreate = d3d11 ? (void*)GetProcAddress(d3d11, "D3D11CreateDevice") : nullptr;
    void* realCreateSc = d3d11 ? (void*)GetProcAddress(d3d11, "D3D11CreateDeviceAndSwapChain") : nullptr;
    int patched = 0;
    for (IMAGE_IMPORT_DESCRIPTOR* d = (IMAGE_IMPORT_DESCRIPTOR*)(base + dir.VirtualAddress); d->Name; d++)
    {
        const char* name = (const char*)(base + d->Name);
        if (_stricmp(name, "d3d11.dll") != 0)
            continue;
        IMAGE_THUNK_DATA64* iat = (IMAGE_THUNK_DATA64*)(base + d->FirstThunk);
        IMAGE_THUNK_DATA64* ilt = d->OriginalFirstThunk ? (IMAGE_THUNK_DATA64*)(base + d->OriginalFirstThunk) : nullptr;
        for (int i = 0; iat[i].u1.Function; i++)
        {
            const char* fn = nullptr;
            if (ilt && !(ilt[i].u1.Ordinal & IMAGE_ORDINAL_FLAG64))
                fn = (const char*)((IMAGE_IMPORT_BY_NAME*)(base + ilt[i].u1.AddressOfData))->Name;
            void* cur = (void*)iat[i].u1.Function;
            void* hook = nullptr;
            void** orig = nullptr;
            if ((fn && strcmp(fn, "D3D11CreateDevice") == 0) || (!fn && cur == realCreate))
                hook = (void*)Hk_GameD3D11CreateDevice, orig = (void**)&o_GameD3D11CreateDevice;
            else if ((fn && strcmp(fn, "D3D11CreateDeviceAndSwapChain") == 0) || (!fn && cur == realCreateSc))
                hook = (void*)Hk_GameD3D11CreateDeviceAndSwapChain, orig = (void**)&o_GameD3D11CreateDeviceAndSwapChain;
            if (!hook)
                continue;
            DWORD old;
            if (VirtualProtect(&iat[i].u1.Function, 8, PAGE_READWRITE, &old))
            {
                *orig = cur;
                iat[i].u1.Function = (ULONGLONG)hook;
                VirtualProtect(&iat[i].u1.Function, 8, old, &old);
                patched++;
            }
        }
    }
    LOGI("Feature-level compatibility: %d game import(s) hooked", patched);
}

void ModBootstrap()
{
    if (g_cfg.hookDelayMs > 0)
        Sleep(g_cfg.hookDelayMs);

    HMODULE d3d11 = LoadLibraryW(L"d3d11.dll");
    PFN_D3D11_CREATE_DEVICE create = d3d11 ? (PFN_D3D11_CREATE_DEVICE)GetProcAddress(d3d11, "D3D11CreateDevice") : nullptr;
    if (!create)
    {
        LOGE("d3d11.dll / D3D11CreateDevice not available");
        return;
    }
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    D3D_FEATURE_LEVEL fls[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL got;
    HRESULT hr = create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, fls, 2, D3D11_SDK_VERSION, &dev, &got, &ctx);
    if (FAILED(hr))
        hr = create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, fls + 1, 1, D3D11_SDK_VERSION, &dev, &got, &ctx);
    if (FAILED(hr) || !dev || !ctx)
    {
        LOGE("Temporary D3D11 device creation failed 0x%08lX", hr);
        return;
    }
    // The device vtable is shared by every device of the same class: d3d11.dll's own, or a wrapper's
    // proxy class when ReShade wraps devices (so only call these originals on objects that came through
    // the vtable). Context vtables are per device on current Windows builds, so those are hooked when
    // the game's device shows up.
    s_devVt = *(void***)dev;
    PatchSlot(s_devVt, kDev_CreatePixelShader, (void*)Hk_CreatePixelShader, (void**)&o_CreatePixelShader);
    PatchSlot(s_devVt, kDev_CreateDeferredContext, (void*)Hk_CreateDeferredContext, (void**)&o_CreateDeferredContext);
    PatchSlot(s_devVt, kDev_CreateComputeShader, (void*)Hk_CreateComputeShader, (void**)&o_CreateComputeShader);
    if (g_cfg.menuKey)
        PatchSlot(s_devVt, kDev_CreateRenderTargetView, (void*)Hk_CreateRTV, (void**)&o_CreateRTV);
    if (g_cfg.maskFeatureLevel)
    {
        PatchSlot(s_devVt, 37 /* GetFeatureLevel */, (void*)Hk_GetFeatureLevel, (void**)&o_GetFeatureLevel);
        bool a = ChainExport(d3d11, "D3D11CreateDevice", (void*)Hk_GameD3D11CreateDevice, (void**)&o_GameD3D11CreateDevice);
        bool b = ChainExport(d3d11, "D3D11CreateDeviceAndSwapChain", (void*)Hk_GameD3D11CreateDeviceAndSwapChain,
                             (void**)&o_GameD3D11CreateDeviceAndSwapChain);
        if (!a && !b)
            HookGameImports();
    }
    HMODULE vtMod = nullptr;
    char vtPath[MAX_PATH] = "?";
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCWSTR)s_devVt, &vtMod) &&
        vtMod)
        GetModuleFileNameA(vtMod, vtPath, MAX_PATH);
    LOGI("Device vtable %p (%s) hooked (CreatePixelShader, CreateDeferredContext)", s_devVt, vtPath);
    ctx->Release();
    dev->Release();

    CompileShaders();
    OverlayInit();
    SlCamInit();
    LOGI("Hooks installed, waiting for the game's TAA pass");
}
