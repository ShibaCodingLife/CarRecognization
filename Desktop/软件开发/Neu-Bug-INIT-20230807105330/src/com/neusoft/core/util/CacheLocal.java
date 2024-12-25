package com.neusoft.core.util;

import java.util.HashMap;
import java.util.Map;

import com.neusoft.core.common.conf.Params;

/**
 * @ClassName: CacheLocal
 * @Description: 轻量级的缓存对象容器
 */
public class CacheLocal {

	/**
	 * @Fields _CACHE : 缓存对象容器
	 */
	private static Map<String, Object> _CACHE = new HashMap<String, Object>();

	/**
	 * @Fields _LAST_TIME : 缓存
	 */
	private final static String _LAST_TIME = "_LAST_TIME";

	/**
	 * @Title: getCache
	 * @Description: 读取缓存数据
	 * @param key
	 *            关键字
	 * @return 键对应的对象
	 */
	public static Object getCache(String key) {
		if (key != null) {
			if (_CACHE.containsKey(key)) {
				long optime = (long) _CACHE.get(key + _LAST_TIME);
				if (System.currentTimeMillis() - optime > Params.CACHE_PERIOD) {
					// 缓存数据过期，清空
					_CACHE.remove(key);
					_CACHE.remove(key + _LAST_TIME);
				} else {
					// 读取缓存数据
					return _CACHE.get(key);
				}
			}
		}
		return null;
	}

	/**
	 * @Title: putCache
	 * @Description: 存储缓存数据
	 * @param key
	 *            缓存关键字
	 * @param value
	 *            缓存对象
	 * @return true-成功，false-失败
	 */
	public static boolean putCache(String key, Object value) {
		// 缓存数量检查
		if (_CACHE.size() > Params.CACHE_MAX) {
			_CACHE.clear();
		}
		// 检查数据
		if (key == null || value == null) {
			return false;
		} else {
			// 存储数据，并且记录对应的时间
			_CACHE.put(key, value);
			_CACHE.put(key + _LAST_TIME, System.currentTimeMillis());
			return true;
		}
	}

}
