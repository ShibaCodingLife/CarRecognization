package com.neusoft.core.common;

import java.beans.BeanInfo;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang.builder.ReflectionToStringBuilder;
import org.apache.commons.lang.builder.ToStringStyle;

/**
 * @ClassName: BaseVO
 * @Description: 基础VO类
 */
public class BaseVO extends BaseObject {

	/**
	 * @Title: toString
	 * @Description: VO转字符串
	 * @return Map<String,Object>
	 */
	@Override
	public String toString() {
		return ReflectionToStringBuilder.toString(this,
				ToStringStyle.MULTI_LINE_STYLE);
	}

	/**
	 * @Title: toMap
	 * @Description: VO转换为Map<String,Object>对象，只能转换简单类型的属性
	 * @return Map<String,Object>
	 */
	public Map<String, Object> toMap() {
		Map<String, Object> map = new HashMap<String, Object>();
		try {
			BeanInfo beanInfo = Introspector.getBeanInfo(this.getClass());
			PropertyDescriptor[] propertyDescriptors = beanInfo
					.getPropertyDescriptors();
			for (PropertyDescriptor property : propertyDescriptors) {
				String key = property.getName();

				// 过滤class属性
				if (!key.equals("class")) {
					// 得到property对应的getter方法
					Method getter = property.getReadMethod();
					Object value = getter.invoke(this);
					if (value != null)
						map.put(key, value);
					else
						log.warn("Value of " + key
								+ " property is null object,so ignore!");
				}
			}
		} catch (Exception e) {
			log.error(e);
		}

		return map;
	}

	/**
	 * @Title: toMapString
	 * @Description: VO转换为Map<String,String>对象，只能转换简单类型的属性
	 * @return Map<String, String>
	 */
	public Map<String, String> toMapString() {
		Map<String, String> map = new HashMap<String, String>();
		try {
			BeanInfo beanInfo = Introspector.getBeanInfo(this.getClass());
			PropertyDescriptor[] propertyDescriptors = beanInfo
					.getPropertyDescriptors();
			for (PropertyDescriptor property : propertyDescriptors) {
				String key = property.getName();

				// 过滤class属性
				if (!key.equals("class")) {
					// 得到property对应的getter方法
					Method getter = property.getReadMethod();
					Object value = getter.invoke(this);
					if (value == null)
						log.warn("Value of " + key
								+ " property is null object,so ignore!");
					else if (value instanceof String)
						map.put(key, (String) value);
					else
						log.warn("Type of " + key
								+ " property is not String,so ignore!");
				}
			}
		} catch (Exception e) {
			log.error(e);
		}

		return map;
	}
}
