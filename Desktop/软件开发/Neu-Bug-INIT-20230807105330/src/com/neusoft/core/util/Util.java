package com.neusoft.core.util;

import java.beans.BeanInfo;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.lang.reflect.Method;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

/**
 * @ClassName: Util
 * @Description: 工具类
 */
public class Util {

	/**
	 * @Title: isEmpty
	 * @Description: Web环境下判断对象是否为空 字符串长度为零被定义为空
	 * @param obj
	 * @return boolean
	 */
	public static boolean isEmpty(Object obj) {
		if (obj == null)
			return true;
		if (obj instanceof String)
			return ((String) obj).trim().length() == 0;
		return false;
	}

	/**
	 * @Title: isNotEmpty
	 * @Description: 对象是否不为空
	 * @param obj
	 * @return boolean
	 */
	public static boolean isNotEmpty(Object obj) {
		return !isEmpty(obj);
	}

	/**
	 * @Title: notNull
	 * @Description: 将String以非null形式返回，如果String为null，则返回一个空字符串""。
	 * @param str
	 * @return String
	 */
	public static String notNull(String str) {
		return str == null || "null".equalsIgnoreCase(str) ? "" : str;
	}

	/**
	 * @Title: parseInt
	 * @Description: 将String类型转化为int类型，转化失败则为0。
	 * @param value
	 * @return int
	 */
	public static int parseInt(String value) {
		int result = 0;
		try {
			result = Integer.parseInt(value);
		} catch (Exception ex) {
		}
		return result;
	}

	/**
	 * @Title: getLocalDate
	 * @Description: 按yyyy-MM-dd格式返回日期
	 * @return yyyy-MM-dd格式日期
	 */
	public static String getLocalDate() {
		return getDate("yyyy-MM-dd");
	}

	/**
	 * @Title: getDate
	 * @Description: 按指定格式返回日期串
	 * @param format
	 *            格式
	 * @return String
	 */
	public static String getDate(String format) {
		return new SimpleDateFormat(format).format(new Date());
	}

	/**
	 * @Title: jsonp
	 * @Description: "jsoncallback({'success':'true','a':'1'})"
	 * @param json
	 * @return "jsoncallback({'success':'true','a':'1'})"
	 */
	public static String jsonp(String json) {
		return jsonp("jsoncallback", json);
	}

	/**
	 * @Title: jsonp
	 * @Description: "prefix({'success':'true','a':'1'})"
	 * @param prefix
	 * @param json
	 * @return "prefix({'success':'true','a':'1'})"
	 */
	public static String jsonp(String prefix, String json) {
		return prefix + "(" + json + ")";
	}

	/**
	 * @Title: convBeanToMap
	 * @Description: JavaBean转为Map
	 * @param bean
	 *            JavaBean
	 * @param ignoreNull
	 *            是否忽略空
	 * @return Map<String, Object>
	 */
	@SuppressWarnings("rawtypes")
	public static Map<String, Object> convBeanToMap(Object bean,
			boolean ignoreNull) {
		Class type = bean.getClass();
		Map<String, Object> r = new HashMap<String, Object>();
		try {
			BeanInfo beanInfo = Introspector.getBeanInfo(type);
			PropertyDescriptor[] propertyDescriptors = beanInfo
					.getPropertyDescriptors();
			for (int i = 0; i < propertyDescriptors.length; i++) {
				PropertyDescriptor descriptor = propertyDescriptors[i];
				String propertyName = descriptor.getName();
				if (!propertyName.equals("class")) {
					Method readMethod = descriptor.getReadMethod();
					Object result = readMethod.invoke(bean, new Object[0]);
					if (result != null) {
						r.put(propertyName, result);
					} else if (!ignoreNull)
						r.put(propertyName, null);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return r;
	}

	/**
	 * @Title: convBeanToMap
	 * @Description: JavaBean转为Map 忽略空
	 * @param bean
	 *            JavaBean
	 * @return Map<String, Object>
	 */
	public static Map<String, Object> convBeanToMap(Object bean) {
		return convBeanToMap(bean, true);
	}

	/**
	 * @Title: bytesToHex
	 * @Description: byte[]转十六进制
	 * @param bArray
	 *            byte[]
	 * @param upper
	 *            是否转为大写
	 * @return String
	 */
	public static String bytesToHex(byte[] bArray, boolean upper) {
		StringBuffer sb = new StringBuffer(bArray.length);
		String sTemp;
		for (int i = 0; i < bArray.length; i++) {
			sTemp = Integer.toHexString(0xFF & bArray[i]);
			if (sTemp.length() < 2)
				sb.append(0);
			sb.append(sTemp.toLowerCase());
		}
		return upper ? sb.toString().toUpperCase() : sb.toString();
	}

	/**
	 * @Title: bytesToHex
	 * @Description: byte[]转十六进制 不转大写
	 * @param bArray
	 *            byte[]
	 * @return String
	 */
	public static String bytesToHex(byte[] bArray) {
		return bytesToHex(bArray, false);
	}

}
