import axios from "axios";
import { getIdToken } from "firebase/auth";
import { auth } from "./firebase";

const API = axios.create({ baseURL: "https://your-api.com" });
API.interceptors.request.use(async config => {
  const token = await getIdToken(auth.currentUser);
  config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export const postHistory = data => API.post("/api/history", data);
export const getHistory  = () => API.get("/api/history");
