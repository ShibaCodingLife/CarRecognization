package com.neusoft.core.common.conf;

import com.neusoft.core.util.PCache;

/**
 * @ClassName: Params
 * @Description: 参数常量类
 */
public class Params {

	/**
	 * @Fields ENCODING : 默认字符集
	 */
	public static final String ENCODING = "UTF-8";

	/**
	 * @Fields CONTENT_TYPE_UTF8_JSON : JSON默认字符集
	 */
	public static final String CONTENT_TYPE_UTF8_JSON = "text/json;charset=UTF-8";

	/**
	 * @Fields CACHE_MAX : 简单缓存-最大缓存对象数量(因为包含缓存时间标记位，实际上是设定数量的1/2)
	 */
	public static final int CACHE_MAX = Integer.parseInt(PCache.getHM().get(
			"cache.local.max"));

	/**
	 * @Fields CACHE_PERIOD : 简单缓存-最大缓存有效时间
	 */
	public static final long CACHE_PERIOD = Long.parseLong(PCache.getHM().get(
			"cache.local.period"));

	/**
	 * @Fields RET_CODE : 返回代码KEY常量
	 */
	public static final String RET_CODE = "RET_CODE";

	/**
	 * @Fields RET_MSG : 返回消息KEY常量
	 */
	public static final String RET_MSG = "RET_MSG";

	/**
	 * @Fields RET_OBJ : 返回对象KEY常量
	 */
	public static final String RET_OBJ = "RET_OBJ";

	/**
	 * @Fields RET_CODE_SUCCESS : 返回成功代码
	 */
	public static final String RET_CODE_SUCCESS = "success";

	/**
	 * @Fields RET_CODE_ERROR : 返回失败代码
	 */
	public static final String RET_CODE_ERROR = "error";

	/**
	 * @Fields SESSION_ATTR_KEY_USER : session中保存登录用户信息关键字
	 */
	public static final String SESSION_ATTR_KEY_USER = "user";

}
