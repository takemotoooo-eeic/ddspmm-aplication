import type { AxiosRequestConfig } from "axios";
import axios from "axios";

export const instance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  headers: {
    Accept: "application/json",
    "Content-Type": "application/json; charset=utf-8",
  },
});

export const useMutator = <T = unknown>(config: AxiosRequestConfig): Promise<T> => {
  // to avoid putting "[]" to query params of array type
  config.paramsSerializer = {
    indexes: null,
  };

  return instance(config).then(({ data }) => data);
};
