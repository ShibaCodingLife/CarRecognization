package com.neusoft.core.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.gson.ExclusionStrategy;
import com.google.gson.FieldAttributes;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.neusoft.core.common.conf.Params;

/**
 * @ClassName: JsonUtil
 * @Description: json工具类
 */
public class JsonUtil {

	/**
	 * @Fields myExclusionStrategy : 定义GSON策略
	 */
	private static ExclusionStrategy myExclusionStrategy = new ExclusionStrategy() {
		@Override
		public boolean shouldSkipField(FieldAttributes arg0) {
			return arg0.getName().startsWith("log");
		}

		@Override
		public boolean shouldSkipClass(Class<?> arg0) {
			return false;
		}
	};

	/**
	 * @Fields g : json操作工具
	 */
	private final static Gson g = new GsonBuilder().setExclusionStrategies(
			myExclusionStrategy).create();

	/**
	 * @Title: getGson
	 * @Description: 获取gjson
	 * @return gjson
	 */
	public static Gson getGson() {
		return g;
	}

	/**
	 * @Title: main
	 * @Description: 使用方法
	 * @param args
	 */
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public static void main(String[] args) {
		Map<String, Object> r = new HashMap<String, Object>();
		List<Map> l = new ArrayList<Map>();
		Map<String, Object> e = new HashMap<String, Object>();
		e.put("id", 12313);
		e.put("name", "afadfdf");
		l.add(e);
		r.put("list", l);
		r.put("num", "212");
		r.put("ret", "success");
		String j = g.toJson(r);
		System.out.println(j);
		Map<String, Object> jm = g.fromJson(j, Map.class);
		jm = g.fromJson(jm.toString(), Map.class);
		System.out.println(jm);
		System.out.println(Params.RET_CODE_SUCCESS.equals(jm.get("ret")));
	}
}
