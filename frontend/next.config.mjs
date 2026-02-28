/** @type {import('next').NextConfig} */
const envAllowedDevOrigins = (process.env.NEXT_ALLOWED_DEV_ORIGINS ?? "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

const nextConfig = {
  reactStrictMode: true,
  allowedDevOrigins: [
    "localhost",
    "127.0.0.1",
    "10.0.0.95",
    ...envAllowedDevOrigins,
  ],
};

export default nextConfig;
