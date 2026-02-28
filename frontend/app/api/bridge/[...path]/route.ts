import { NextRequest } from "next/server";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function resolveApiBase(request: NextRequest): string {
  const headerValue = request.headers.get("x-neuromap-api-url")?.trim();
  const envValue = process.env.NEUROMAP_API_URL?.trim();
  const base = headerValue || envValue || DEFAULT_API_BASE;

  if (!/^https?:\/\//i.test(base)) {
    throw new Error("API URL must start with http:// or https://");
  }
  return base.replace(/\/+$/, "");
}

async function proxy(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
): Promise<Response> {
  try {
    const { path } = await context.params;
    const incomingUrl = new URL(request.url);
    const apiBase = resolveApiBase(request);
    const target = new URL(`${apiBase}/${path.join("/")}`);
    target.search = incomingUrl.search;

    const headers = new Headers(request.headers);
    headers.delete("host");
    headers.delete("content-length");
    headers.delete("x-neuromap-api-url");

    const method = request.method.toUpperCase();
    const hasBody = method !== "GET" && method !== "HEAD";
    const body = hasBody ? await request.arrayBuffer() : undefined;

    const upstream = await fetch(target, {
      method,
      headers,
      body: hasBody ? body : undefined,
      redirect: "manual",
      cache: "no-store",
    });

    const responseHeaders = new Headers(upstream.headers);
    responseHeaders.set("x-neuromap-proxy", "1");
    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Proxy request failed.";
    return Response.json({ detail: message }, { status: 502 });
  }
}

export async function GET(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, context);
}

export async function POST(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, context);
}

export async function PUT(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, context);
}

export async function PATCH(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, context);
}

export async function DELETE(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, context);
}
